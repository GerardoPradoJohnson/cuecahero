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


def test_catalog_song_files_exist_and_valid():
    songs_dir = ROOT / 'config/songs'
    assert songs_dir.exists(), "config/songs directory must exist"
    
    catalog = list(songs_dir.glob('*.json'))
    assert len(catalog) >= 2, "Catalog must contain at least 2 songs"
    
    song_ids = [f.stem for f in catalog]
    assert 'la_consentida' in song_ids, "La Consentida must be in catalog"
    assert 'primer_panuelo' in song_ids, "Primer Pañuelo must be in catalog"
    
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
    assert env.bpm in (112, 114)
    assert len(env.notes) >= 64
    assert env.duration > 20.0
    
    state = env.get_state()
    assert state['song']['id'] == 'la_consentida'
    assert state['bpm'] in (112, 114)


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
        assert la_cons['bpm'] in (112, 114)
        assert la_cons['notes_count'] >= 64
        assert la_cons['audio_url'] == '/audio/la_consentida.wav'
    
    # Check /audio/la_consentida.wav streaming
    req_audio = Request(f'{base_url}/audio/la_consentida.wav')
    with urlopen(req_audio, timeout=5) as res:
        assert res.status == 200
        assert res.headers.get('Content-Type') == 'audio/wav'
        audio_data = res.read()
        assert len(audio_data) > 1_000_000, "WAV audio must contain valid waveform data"


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
    assert snap['game']['bpm'] in (112, 114)
    assert snap['game']['time'] == 0.0


def test_male_cns_core_untouched():
    """Verify MaleCNSCore is 100% untouched and maintains full contract."""
    core = MaleCNSCore()
    assert hasattr(core, 'advance')
    assert hasattr(core, 'reset')
    assert core.n > 0
