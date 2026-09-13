"""Keep upstream pristine; pin checkout and install a CPU baseline."""
import json
import platform
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def run(*args, **kw):
    subprocess.run(args, check=True, **kw)
def main():
    system, machine = platform.system(), platform.machine().lower()
    if (system, machine) not in {('Darwin','x86_64'),('Darwin','arm64'),('Windows','amd64'),('Windows','x86_64')}:
        raise SystemExit(f'Unsupported platform: {system} {machine}')
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('Python 3.11 required')
    print(f'{system} {machine}; backend=CPU; CUDA not required', flush=True)
    pin = json.loads((ROOT / 'config/doomfly.lock.json').read_text())
    upstream = ROOT / 'external/doomfly'
    if not upstream.exists():
        run('git', 'clone', pin['url'], str(upstream))
    if subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'], text=True).strip():
        raise SystemExit('Upstream checkout has changes; refusing to use modified sources.')
    actual = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip()
    if actual != pin['commit']:
        dirty = subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'], text=True)
        if dirty:
            raise SystemExit('Upstream checkout has local changes; refusing to overwrite.')
        run('git','-C',str(upstream),'fetch','origin',pin['commit'])
        run('git','-C',str(upstream),'checkout','--detach',pin['commit'])
    run(sys.executable,'-m','pip','install','pip==25.3')
    run(sys.executable,'-m','pip','install','-r',str(ROOT/'requirements.txt'),'-c',str(ROOT/'config/python-constraints.txt'),'--build-constraint',str(upstream/'neural-build-constraints.txt'))
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/environment-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
    try:
        run(sys.executable,str(ROOT/'scripts/build_kernel.py'))
    except subprocess.CalledProcessError:
        print('Optional native build failed; CPU/Numba remains available. See compiler output above.', flush=True)
if __name__ == '__main__':
    main()
