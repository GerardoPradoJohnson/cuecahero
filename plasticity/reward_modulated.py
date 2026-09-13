"""Bounded reward-modulated Hebbian rule on identified existing graph edges.

This is an engineering hypothesis. Reward is an external scalar modulator; it is
not a biological dopamine simulation. The immutable baseline graph stays on disk.
"""
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class RewardModulatedPlasticity:
    enabled = True

    def __init__(self, eta=.002, eligibility_tau_ms=750., minimum_fraction=.98,
                 maximum_fraction=1.02):
        values = [eta, eligibility_tau_ms, minimum_fraction, maximum_fraction]
        if not np.isfinite(values).all() or eta <= 0 or eligibility_tau_ms <= 0 or not 0 < minimum_fraction <= 1 <= maximum_fraction or maximum_fraction <= minimum_fraction:
            raise ValueError('Invalid reward-modulated plasticity parameters')
        self.eta = float(eta)
        self.tau_ms = float(eligibility_tau_ms)
        self.minimum_fraction = float(minimum_fraction)
        self.maximum_fraction = float(maximum_fraction)
        self.bound = False

    def bind(self, neural_state, presynaptic, postsynaptic, graph_sha256):
        state = neural_state.state
        pre = np.unique(np.asarray(presynaptic, dtype=np.int32))
        post_mask = np.zeros(state.n, dtype=bool)
        post_mask[np.asarray(postsynaptic, dtype=np.int32)] = True
        edges, edge_pre = [], []
        for source in pre:
            begin, end = state.ptr[source:source+2]
            chosen = np.flatnonzero(post_mask[state.post[begin:end]]) + begin
            edges.extend(chosen.tolist())
            edge_pre.extend([int(source)] * len(chosen))
        self.edges = np.asarray(edges, dtype=np.int64)
        self.pre = np.asarray(edge_pre, dtype=np.int32)
        if not len(self.edges):
            raise ValueError('No existing edges connect the sensory and readout populations')
        self.post = state.post[self.edges].copy()
        self.baseline = state.weight[self.edges].copy()
        if np.any(self.baseline == 0) or not np.isfinite(self.baseline).all():
            raise ValueError('Plastic edges need finite nonzero baseline weights')
        self.eligibility = np.zeros(len(self.edges), dtype=np.float64)
        self.graph_sha256 = graph_sha256
        self.edge_sha256 = digest(self.edges)
        self.last_reward = 0.
        self.updates = 0
        self.bound = True

    def apply(self, neural_state, reward):
        if not self.bound:
            raise RuntimeError('Plasticity must be bound to a verified brain')
        if reward.source != 'cueca_hero' or not math.isfinite(reward.magnitude):
            raise ValueError('Invalid reward signal for plasticity')
        counts = neural_state.state.counts
        decay = math.exp(-10. / self.tau_ms)
        # Binary coincidence avoids rewarding tonic firing magnitude repeatedly.
        coincidence = ((counts[self.pre] > 0) & (counts[self.post] > 0)).astype(float)
        self.eligibility *= decay
        self.eligibility += coincidence
        self.last_reward = float(reward.magnitude)
        if reward.magnitude == 0:
            return
        normalized = self.eligibility / max(1., float(self.eligibility.max()))
        state = neural_state.state
        fraction = state.weight[self.edges] / self.baseline
        # Positive feedback reinforces recent coactivity; negative feedback weakens it.
        fraction += self.eta * reward.magnitude * normalized
        fraction = np.clip(fraction, self.minimum_fraction, self.maximum_fraction)
        state.weight[self.edges] = self.baseline * fraction
        self.updates += 1

    def status(self, neural_state=None):
        if not self.bound:
            return {'enabled':True, 'model':'reward_modulated_hebbian_v1', 'plastic_edges':0, 'changed_edges':0}
        weight = neural_state.state.weight[self.edges] if neural_state else self.baseline
        fraction = weight / self.baseline
        return {'enabled':True, 'model':'reward_modulated_hebbian_v1',
                'plastic_edges':len(self.edges),
                'changed_edges':int(np.count_nonzero(weight != self.baseline)),
                'mean_fraction':float(fraction.mean()),
                'minimum_fraction':float(fraction.min()),
                'maximum_fraction':float(fraction.max()),
                'updates':self.updates, 'last_reward':self.last_reward,
                'edge_sha256':self.edge_sha256}

    def erase(self, neural_state):
        neural_state.state.weight[self.edges] = self.baseline
        self.eligibility.fill(0)
        self.updates = 0
        self.last_reward = 0.

    def checkpoint(self, neural_state, path):
        status = self.status(neural_state)
        metadata = {'schema':1, 'model':'reward_modulated_hebbian_v1',
                    'graph_sha256':self.graph_sha256, 'edge_sha256':self.edge_sha256,
                    'eta':self.eta, 'eligibility_tau_ms':self.tau_ms,
                    'minimum_fraction':self.minimum_fraction,
                    'maximum_fraction':self.maximum_fraction,
                    'updates':self.updates, 'last_reward':self.last_reward,
                    'status':status}
        weights = neural_state.state.weight[self.edges].copy()
        checksum = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()+weights.tobytes()+self.eligibility.tobytes()).hexdigest()
        buffer = io.BytesIO()
        np.savez_compressed(buffer, metadata=json.dumps(metadata), weights=weights,
                            eligibility=self.eligibility, checksum=checksum)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.memory-', delete=False) as stream:
                temporary = Path(stream.name); stream.write(buffer.getvalue()); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            if temporary: temporary.unlink(missing_ok=True)
        return checksum

    def restore(self, neural_state, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            weights = data['weights'].copy(); eligibility = data['eligibility'].copy()
            checksum = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()+weights.tobytes()+eligibility.tobytes()).hexdigest()
            if checksum != str(data['checksum']): raise ValueError('Plasticity checkpoint checksum mismatch')
        expected = {'schema':1, 'model':'reward_modulated_hebbian_v1',
                    'graph_sha256':self.graph_sha256, 'edge_sha256':self.edge_sha256,
                    'eta':self.eta, 'eligibility_tau_ms':self.tau_ms,
                    'minimum_fraction':self.minimum_fraction, 'maximum_fraction':self.maximum_fraction}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError('Plasticity checkpoint provenance mismatch')
        if weights.shape != self.baseline.shape or eligibility.shape != self.eligibility.shape or not np.isfinite(weights).all() or not np.isfinite(eligibility).all() or np.any(eligibility < 0):
            raise ValueError('Invalid plasticity checkpoint arrays')
        fraction = weights / self.baseline
        if np.any(fraction < self.minimum_fraction) or np.any(fraction > self.maximum_fraction):
            raise ValueError('Checkpoint weights exceed configured bounds')
        neural_state.state.weight[self.edges] = weights
        self.eligibility[:] = eligibility
        self.updates = int(metadata['updates']); self.last_reward = float(metadata['last_reward'])
