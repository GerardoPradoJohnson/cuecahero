"""Local checkpoint catalogue and provenance-checked evaluation recordings."""
import hashlib
import json
from pathlib import Path
from core.paths import ROOT


def resolve_checkpoint(relative):
    root = (ROOT / 'outputs/training').resolve()
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(root) or path.suffix != '.npz' or not path.is_file():
        raise ValueError('Checkpoint unavailable')
    return path


def catalogue():
    rows = []
    root = ROOT / 'outputs/training'
    for summary in sorted(root.glob('*/evaluation.json')):
        data = json.loads(summary.read_text(encoding='utf-8'))
        path = summary.parent / data['checkpoint']
        if not path.is_file():
            continue
        playback = summary.parent / data.get('playback', '') if data.get('playback') else None
        rows.append(dict(path=str(path.relative_to(ROOT)).replace('\\', '/'),
                         name=data['name'], generation=data['generation'],
                         metrics=data['metrics'], method=data['method'],
                         playback_available=bool(playback and playback.is_file())))
    jobs = []
    for status in sorted(root.glob('*/status.json')):
        data = json.loads(status.read_text(encoding='utf-8'))
        if data.get('stage') != 'failed':
            jobs.append(dict(run=status.parent.name, **data))
    return dict(checkpoints=rows, jobs=jobs)


def recording_path(relative):
    checkpoint = resolve_checkpoint(relative)
    summary = json.loads((checkpoint.parent / 'evaluation.json').read_text(encoding='utf-8'))
    if checkpoint.name != summary['checkpoint']:
        raise ValueError('No evaluated recording for this checkpoint')
    if not summary.get('playback'):
        raise ValueError('This checkpoint has no fixed-song recording')
    path = checkpoint.parent / summary['playback']
    for file, expected in [(checkpoint, summary['checkpoint_sha256']),
                           (path, summary['playback_sha256']),
                           (ROOT/'config/songs/la_consentida.json', summary['song_sha256']),
                           (ROOT/'frontend/audio/consentida.mp3', summary['audio_sha256'])]:
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Recording, checkpoint or song changed; evaluate again')
    return path
