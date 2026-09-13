"""Match legacy NumPy archives across Windows and Unix without changing data."""
import hashlib
import struct
import zipfile
from pathlib import Path


def calibration_graph_matches(path, expected, actual):
    if actual == expected:
        return True
    # np.savez records the writer OS in each central-directory entry. The
    # original calibration was written on Unix. Normalize only that byte;
    # every array, dtype, ordering and remaining archive byte stays protected.
    data = bytearray(Path(path).read_bytes())
    with zipfile.ZipFile(path) as archive:
        position = archive.start_dir
        for _ in archive.infolist():
            if data[position:position + 4] != b'PK\x01\x02':
                return False
            data[position + 5] = 3
            name, extra, comment = struct.unpack_from('<HHH', data, position + 28)
            position += 46 + name + extra + comment
    return hashlib.sha256(data).hexdigest() == expected
