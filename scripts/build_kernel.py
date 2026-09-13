"""Build upstream C++ unchanged; optional native acceleration of CPU baseline."""
import ctypes
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from data_common import ROOT

def main():
    source = ROOT / 'external/doomfly/doom/kernel.cpp'
    directory = ROOT / 'build'
    directory.mkdir(exist_ok=True)
    windows = platform.system() == 'Windows'
    out = directory / ('neural.dll' if windows else 'libneural.dylib')
    clang = shutil.which('clang++')
    msvc = shutil.which('cl') if windows else None
    if not clang and not msvc:
        print('No C++ compiler found; CPU Numba backend remains available.')
        return
    tmp = out.with_name(out.stem + '.partial' + out.suffix)
    if msvc:
        command = [msvc,'/LD','/O2','/std:c++17',str(source),'/link','/EXPORT:neural_advance',f'/OUT:{tmp}']
        subprocess.run(command, check=True, cwd=directory)
    else:
        command = [clang,'-O3','-std=c++17','-shared',str(source),'-o',str(tmp)]
        if not windows:
            command.insert(1, '-fPIC')
            if platform.system() == 'Darwin':
                universal_command = [clang, '-arch', 'arm64', '-arch', 'x86_64', '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-o', str(tmp)]
                res = subprocess.run(universal_command, cwd=directory, capture_output=True)
                if res.returncode == 0:
                    command = universal_command
                else:
                    subprocess.run(command, check=True, cwd=directory)
            else:
                subprocess.run(command, check=True, cwd=directory)
        else:
            command.append('-Wl,--export-all-symbols')
            subprocess.run(command, check=True, cwd=directory)
    record = {'kernel_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'binary_sha256': hashlib.sha256(tmp.read_bytes()).hexdigest(),
              'model_revision': 'lif-r2-refractory-write-protection', 'command': command}
    # Confirm exported C ABI before marking build successful.
    subprocess.run([sys.executable, '-c',
                    'import ctypes,sys; getattr(ctypes.CDLL(sys.argv[1]), "neural_advance")',
                    str(tmp)], check=True)
    tmp.replace(out)
    out.with_suffix(out.suffix + '.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Built CPU kernel: {out}')
if __name__ == '__main__':
    main()
