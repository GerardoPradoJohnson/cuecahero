import hashlib
import zipfile

from core.graph_identity import calibration_graph_matches


def archive(path, system, payload):
    entry = zipfile.ZipInfo('neurons.npy')
    entry.create_system = system
    with zipfile.ZipFile(path, 'w') as output:
        output.writestr(entry, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_platform_metadata_does_not_invalidate_calibration(tmp_path):
    unix = archive(tmp_path / 'unix.npz', 3, b'original neurons')
    windows_path = tmp_path / 'windows.npz'
    windows = archive(windows_path, 0, b'original neurons')
    assert windows != unix
    assert calibration_graph_matches(windows_path, unix, windows)
    assert calibration_graph_matches(windows_path, windows, windows)


def test_changed_neurons_still_rejected(tmp_path):
    expected = archive(tmp_path / 'unix.npz', 3, b'original neurons')
    path = tmp_path / 'changed.npz'
    actual = archive(path, 0, b'changed neurons')
    assert not calibration_graph_matches(path, expected, actual)
