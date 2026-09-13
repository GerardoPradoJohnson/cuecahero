"""Real-connectome CPU smoke benchmark, no environment or fabricated dataset."""
import argparse
import json
import os
import platform
import sys
import time
from data_common import ROOT, DATA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['auto','cpu','native'], default='auto')
    parser.add_argument('--duration-ms', type=float, default=1.0)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'external/doomfly'))
    from doom.engine import Brain
    backend = 'cpu'
    cls = Brain
    library = ROOT / 'build' / ('neural.dll' if platform.system() == 'Windows' else 'libneural.dylib')
    if args.backend != 'cpu' and library.exists():
        os.environ['DOOM_KERNEL_PATH'] = str(library)
        try:
            from doom.native import NativeBrain
            cls, backend = NativeBrain, 'native-cpu'
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            if args.backend == 'native':
                raise
            print(f'Native backend unavailable: {error}; falling back to CPU', file=sys.stderr)
    elif args.backend == 'native':
        raise RuntimeError('Native library missing. Run scripts/build_kernel.py')
    import numpy as np
    brain = cls(DATA / 'outputs/doom/malecns_v1/graph.npz')
    start = time.perf_counter()
    counts, _ = brain.step(np.zeros(len(brain.retina)), args.duration_ms, lamina_bias=0)
    result = {'backend':backend, 'neurons':brain.n,'edges':len(brain.post),
              'neural_ms':brain.sim_ms, 'wall_seconds':time.perf_counter()-start,
              'spikes':int(counts.sum()),'stimulus':'zero current; no environment',
              'note':'First CPU call includes JIT compilation. Not biological validation.'}
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/smoke.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
