"""Tiny explicitly synthetic transfer fixtures; never substituted for MaleCNS."""
import hashlib
import importlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import download_malecns as downloader

def configure(tmp_path, monkeypatch):
    payload = b'explicit-test-fixture'
    entry = {'bytes':len(payload), 'sha256':hashlib.sha256(payload).hexdigest(), 'url':'https://example.invalid/source.feather'}
    target = tmp_path/'target'
    target.mkdir()
    monkeypatch.setattr(downloader, 'TARGET', target)
    monkeypatch.setattr(downloader, 'LOCK', {'test.feather':entry})
    monkeypatch.setattr(downloader, 'ROOT', tmp_path)
    monkeypatch.setattr(downloader, 'verify', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['download_malecns.py'])
    return target, payload

def test_verified_download_and_rerun_avoids_network(tmp_path, monkeypatch):
    target, payload = configure(tmp_path, monkeypatch)
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(payload))
    downloader.main()
    assert (target/'test.feather').read_bytes() == payload
    def forbidden(*a, **k):
        raise AssertionError('Network called on verified rerun')
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', forbidden)
    downloader.main()

def test_bad_download_never_promoted(tmp_path, monkeypatch):
    import pytest
    target, _ = configure(tmp_path, monkeypatch)
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'corrupt'))
    with pytest.raises(RuntimeError, match='Checksum mismatch'):
        downloader.main()
    assert not (target/'test.feather').exists()

def test_existing_corruption_preserved(tmp_path, monkeypatch):
    import pytest
    target, _ = configure(tmp_path, monkeypatch)
    (target/'test.feather').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='Existing file is invalid'):
        downloader.main()
    assert (target/'test.feather').read_bytes() == b'corrupt'
