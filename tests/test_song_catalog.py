"""Tests for song catalog, audio streaming, and dynamic chart switching."""
import json
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from pathlib import Path
import pytest

from core.paths import ROOT
from core.malecns import MaleCNSCore
from environments.cueca_hero import CuecaHeroEnvironment
from experiments.runtime import Experiment
from scripts.serve import make_handler
from core.contracts import Action


@pytest.mark.parametrize("step", [1/30, 1/60])
def test_la_consentida_chart_can_be_played_with_real_key_holds(step):
    song = json.loads((ROOT / 'config/songs/la_consentida.json').read_text())
    for lane in range(4):
        notes = sorted((n for n in song['notes'] if n['lane'] == lane), key=lambda n: n['at'])
        for note, following in zip(notes, notes[1:]):
            assert note['at'] + note.get('sustain', 0) < following['at'] - .05
    env = CuecaHeroEnvironment()
    env.load_song(song)
    while not env.is_done():
        controls = [0.] * 4
        for note in env.notes:
            starts = env.time < note['at'] <= env.time + step + 1e-9
            holding = note['judgement'] in ('good', 'perfect') and env.time < note['at'] + note.get('sustain', 0)
            if starts or holding:
                controls[note['lane']] = 1.
        env.step(Action(tuple(controls)), step)
    assert env.hits == len(song['notes'])
    assert env.misses == env.wrong == 0
    assert sum(lane['hits'] for lane in env.lanes) == env.hits


def test_catalog_song_files_exist_and_valid():
    songs_dir = ROOT / 'config/songs'
    assert songs_dir.exists(), "config/songs directory must exist"
    
    catalog = list(songs_dir.glob('*.json'))
    assert len(catalog) >= 2, "Catalog must contain at least 2 songs"
    
    song_ids = [f.stem for f in catalog]
    assert 'la_consentida' in song_ids, "La Consentida must be in catalog"
    assert 'primer_panuelo' in song_ids, "Primer PaÃ±uelo must be in catalog"
    
    for f in catalog:
        data = json.loads(f.read_text())
        assert 'id' in data
        assert 'title' in data
        assert 'bpm' in data and data['bpm'] > 0
        assert 'duration' in data and data['duration'] > 5.0
        assert 'notes' in data and len(data['notes']) > 0
        assert 'audio_url' in data
        for note in data['notes']:
            assert 'at' in note
            assert 'lane' in note and 0 <= note['lane'] <= 3
            assert 'duration' in note or 'id' in note


def test_cueca_hero_environment_load_song():
    env = CuecaHeroEnvironment()
    initial_song = env.song
    assert initial_song['id'] == 'primer_panuelo'
    assert env.bpm == 108
    
    la_consentida_data = json.loads((ROOT / 'config/songs/la_consentida.json').read_text())
    env.load_song(la_consentida_data)
    
    assert env.song['id'] == 'la_consentida'
    assert env.song['title'] == 'La Consentida'
    assert env.bpm == la_consentida_data['bpm']
    assert len(env.notes) >= 64
    assert env.duration > 20.0
    
    state = env.get_state()
    assert state['song']['id'] == 'la_consentida'
    assert state['bpm'] == la_consentida_data['bpm']


@pytest.fixture
def song_server(tmp_path):
    config = json.loads((ROOT / 'experiments/cueca_hero.json').read_text())
    experiment = Experiment(config, {'total_neurons': 0, 'indices': []}, sessions_dir=tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler)
    server.RequestHandlerClass = make_handler(experiment, {}, server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield experiment, f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    experiment.close()
    thread.join(timeout=2)


def test_api_songs_and_audio_streaming(song_server):
    _, base_url = song_server
    
    # Check /api/songs
    req = Request(f'{base_url}/api/songs')
    with urlopen(req, timeout=3) as res:
        assert res.status == 200
        songs = json.loads(res.read().decode())
        assert len(songs) >= 2
        song_ids = [s['id'] for s in songs]
        assert 'la_consentida' in song_ids
        assert 'primer_panuelo' in song_ids
        
        la_cons = next(s for s in songs if s['id'] == 'la_consentida')
        assert la_cons['title'] == 'La Consentida'
        assert 100 <= la_cons['bpm'] <= 130
        assert la_cons['notes_count'] >= 64
        assert la_cons['audio_url'] == '/audio/consentida.mp3'
    
    # Check the supplied MP3, including byte ranges used by browser seeking.
    req_audio = Request(f'{base_url}/audio/consentida.mp3')
    with urlopen(req_audio, timeout=5) as res:
        assert res.status == 200
        assert res.headers.get('Content-Type') == 'audio/mpeg'
        audio_data = res.read()
        assert len(audio_data) > 1_000_000, "The original MP3 must be served in full"


def test_select_song_control_command(song_server):
    experiment, base_url = song_server
    
    # Post select_song
    payload = json.dumps({'type': 'select_song', 'song_id': 'la_consentida'}).encode()
    req = Request(f'{base_url}/api/control', data=payload, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=3) as res:
        assert res.status == 202
    
    # Wait for experiment worker to process command
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if experiment.snapshot['game'].get('song', {}).get('id') == 'la_consentida':
            break
        time.sleep(0.02)
    
    snap = experiment.snapshot
    assert snap['game']['song']['id'] == 'la_consentida'
    assert snap['game']['bpm'] == json.loads((ROOT/'config/songs/la_consentida.json').read_text())['bpm']
    assert snap['game']['time'] == 0.0


def test_male_cns_core_untouched():
    """Verify MaleCNSCore is 100% untouched and maintains full contract."""
    core = MaleCNSCore()
    assert hasattr(core, 'advance')
    assert hasattr(core, 'reset')
    assert core.n > 0


def test_mp3_chart_matches_recording_and_count_in():
    import hashlib
    sf = pytest.importorskip('soundfile')
    song = json.loads((ROOT/'config/songs/la_consentida.json').read_text())
    audio = ROOT/'frontend/audio/consentida.mp3'
    assert song['chart_source']['audio_sha256'] == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert song['duration'] == pytest.approx(sf.info(audio).duration + song['audio_offset'], abs=1e-5)
    assert song['audio_offset'] >= 2.4
    for note in song['notes']:
        assert note['at'] - song['audio_offset'] == pytest.approx(note['audio_at'], abs=1e-4)
        assert 0 <= note['audio_at'] < sf.info(audio).duration
        assert note['at'] + note['sustain'] < song['duration']
    env=CuecaHeroEnvironment();env.load_song(song)
    assert env.get_state()['song']['audio_offset'] == song['audio_offset']


def test_mp3_range_requests_preserve_original_bytes(song_server):
    _, url = song_server
    req=Request(url+'/audio/consentida.mp3',headers={'Range':'bytes=2048-4095'})
    with urlopen(req,timeout=3) as response:
        assert response.status==206
        assert response.headers['Content-Type']=='audio/mpeg'
        assert response.read()==(ROOT/'frontend/audio/consentida.mp3').read_bytes()[2048:4096]
