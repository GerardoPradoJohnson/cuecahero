"""Run upstream independent oracle against CPU and native engines."""
import os
import platform
import subprocess
import sys
from data_common import ROOT
upstream = ROOT/'external/doomfly'
env = dict(os.environ)
env['PYTHONPATH'] = str(upstream)
lib = ROOT/'build'/('neural.dll' if platform.system() == 'Windows' else 'libneural.dylib')
env['DOOM_KERNEL_PATH'] = str(lib)
if not lib.exists():
    raise SystemExit('Build native library to run the CPU/native comparison oracle.')
subprocess.run([sys.executable,'-m','pytest',str(upstream/'tests/test_doom_reference.py'),'-q'], env=env, cwd=ROOT, check=True)
