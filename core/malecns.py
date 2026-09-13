"""General current-in/activity-out adapter; unchanged DOOMFLY numerical kernels.

No environment, reward, retina transform, decoder, or game policy lives here.
"""
import ctypes as C
import hashlib
import json
import math
import platform
import sys
import numpy as np
from .contracts import NeuralStimulus, NeuralActivity
from .paths import ROOT, GRAPH

class MaleCNSCore:
    def __init__(self, graph=GRAPH, backend='auto'):
        if backend not in ('auto', 'cpu', 'native'):
            raise ValueError('Unknown backend')
        sys.path.insert(0, str(ROOT / 'external/doomfly'))
        from doom.engine import Brain, advance
        self.state = Brain(graph)
        self.cpu_advance = advance
        self.backend = 'cpu'
        self.fallback_reason = None
        self.native = None
        if backend != 'cpu':
            try:
                library = ROOT / 'build' / ('neural.dll' if platform.system() == 'Windows' else 'libneural.dylib')
                build = json.loads(library.with_suffix(library.suffix + '.json').read_text())
                source = ROOT / 'external/doomfly/doom/kernel.cpp'
                if build['kernel_source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest() or build['binary_sha256'] != hashlib.sha256(library.read_bytes()).hexdigest():
                    raise ValueError('Native source/binary checksum mismatch')
                self.library = C.CDLL(str(library))
                self.native = self.library.neural_advance
                self.native.argtypes = [C.c_int] + [C.c_void_p] * 11 + [C.c_int, C.c_float] + [C.c_void_p] * 5
                self.native.restype = None
                self.backend = 'native-cpu'
            except (OSError, ValueError, KeyError, AttributeError) as error:
                if backend == 'native':
                    raise
                self.fallback_reason = str(error)
        self.reset()

    @property
    def n(self):
        return self.state.n

    def reset(self):
        s = self.state
        s.v.fill(-52.)
        for name in ('g','drive','refractory','queue','queue_count','counts','active','active_flag','nactive'):
            getattr(s, name).fill(0)
        s.cursor = 0
        s.sim_ms = 0.
        s.total_spikes = 0
        self.previous_drive = np.zeros(s.n, np.float32)
        self.last = np.full(s.n, -1, np.int64)

    def advance(self, stimulus: NeuralStimulus, duration_ms: float) -> NeuralActivity:
        steps = round(duration_ms / .1) if math.isfinite(duration_ms) else 0
        if steps < 1 or not math.isclose(steps * .1, duration_ms, abs_tol=1e-8):
            raise ValueError('Duration must be a positive multiple of 0.1 ms')
        indices, currents = np.asarray(stimulus.indices), np.asarray(stimulus.currents)
        if indices.ndim != 1 or currents.shape != indices.shape or indices.dtype.kind not in 'iu':
            raise ValueError('Stimulus needs one current per integer neuron index')
        if np.any(indices < 0) or np.any(indices >= self.n) or not np.isfinite(currents).all() or np.any(np.abs(currents) > 1e4):
            raise ValueError('Invalid stimulus indices or currents')
        s = self.state
        s.drive.fill(0)
        np.add.at(s.drive, indices, currents)
        if not np.isfinite(s.drive).all() or np.any(np.abs(s.drive) > 1e4):
            raise ValueError('Summed current out of range')
        s.counts.fill(0)
        if self.native is not None:
            clock = np.asarray([s.cursor], dtype=np.int64)
            arrays = [s.ptr,s.post,s.weight,s.v,s.g,s.refractory,s.drive,self.previous_drive,s.queue,s.queue_count,clock]
            self.native(s.n, *[a.ctypes.data for a in arrays], steps, .1,
                        *[a.ctypes.data for a in [s.counts,s.active,s.active_flag,s.nactive,self.last]])
            s.cursor = int(clock[0])
        else:
            # Wake arbitrary stimulated cells, not just the upstream sensory classes.
            wake = np.flatnonzero((s.drive != 0) & (s.active_flag == 0))
            start = int(s.nactive[0])
            s.active[start:start + len(wake)] = wake
            s.active_flag[wake] = 1
            s.nactive[0] += len(wake)
            s.cursor = self.cpu_advance(s.ptr,s.post,s.weight,s.v,s.g,s.refractory,s.drive,
                s.queue,s.queue_count,s.cursor,steps,.1,s.counts,s.active,s.active_flag,s.nactive)
        s.sim_ms = s.cursor * .1
        s.total_spikes += int(s.counts.sum())
        return NeuralActivity(s.counts.copy(), steps * .1, s.sim_ms)
