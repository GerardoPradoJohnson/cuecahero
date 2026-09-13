"""Pinned official inputs; usable with Python's standard library only."""
import hashlib
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'config/malecns.lock.json').read_text())
DATA = Path(os.environ.get('FLYLAB_DATA_DIR', ROOT / 'data')).expanduser().resolve()
TARGET = DATA / 'connectome_data/malecns_v1'
def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == entry['sha256']
def verify():
    failures = [name for name, entry in LOCK.items() if not valid(TARGET / name, entry)]
    if failures:
        raise RuntimeError('Missing or invalid official inputs: ' + ', '.join(failures))
    print('Verified all three official MaleCNS inputs (size + SHA-256).', flush=True)
