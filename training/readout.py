"""Deterministic episodic ridge regression with complete learner checkpoints.

Memory consists of sufficient statistics, feature normalization, model provenance
and consumed episode hashes. Neural fast state is deliberately reset per episode;
these are not mid-simulation brain checkpoints.
"""
import copy
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from decoders.calibrated import CalibratedDecoder


def temporal_features(raw, lags):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Finite nonempty episode features required')
    parts = []
    for lag in lags:
        shifted = np.zeros_like(raw)
        if lag < len(raw):
            shifted[lag:] = raw[:len(raw)-lag]
        parts.append(shifted)
    return np.concatenate(parts, axis=1)


def state_digest(metadata, gram, rhs):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for array in (gram, rhs):
        digest.update(np.ascontiguousarray(array, dtype='<f8').tobytes())
    return digest.hexdigest()


class EpisodicReadoutTrainer:
    def __init__(self, model, ridge=60., positive_weight=5.):
        CalibratedDecoder(model)
        if not np.isfinite([ridge, positive_weight]).all() or ridge <= 0 or positive_weight <= 0:
            raise ValueError('Positive finite training parameters required')
        self.template = copy.deepcopy(model)
        self.ridge, self.positive_weight = float(ridge), float(positive_weight)
        n = len(model['mean']) + 1
        self.gram = np.eye(n) * ridge
        self.gram[-1, -1] = .1
        self.rhs = np.zeros((n, 4))
        self.episodes = []

    def model(self):
        weights = solve(self.gram, self.rhs, assume_a='pos')
        result = copy.deepcopy(self.template)
        result.update(weights=weights[:-1].tolist(), bias=weights[-1].tolist())
        result['training'] = f'Episodic supervised external readout; {len(self.episodes)} episodes. No MaleCNS plasticity.'
        result['training_files_sha256'] = {e['name']: e['sha256'] for e in self.episodes}
        result.pop('threshold_selection', None)
        result.pop('training_wall_seconds', None)
        return result

    def consume(self, path):
        path = Path(path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if any(e['name'] == path.name or e['sha256'] == digest for e in self.episodes):
            raise ValueError('Episode already consumed')
        expected = self.template.get('training_files_sha256', {})
        if path.name not in expected or expected[path.name] != digest:
            raise ValueError('Episode is not in the declared training split or its checksum changed')
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            raw = data['features']
            labels = np.asarray(data['labels'], dtype=float)
            X = temporal_features(raw, self.template.get('lags', [0]))
        if X.shape[1] != len(self.template['mean']) or labels.shape != (len(X), 4) or not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 1)):
            raise ValueError('Invalid episode dimensions or labels')
        Z = np.c_[(X-self.template['mean'])/self.template['scale'], np.ones(len(X))]
        current = self.model()
        prediction = Z @ np.vstack([current['weights'], current['bias']])
        weight = np.where(labels.sum(1) > 0, self.positive_weight, 1.)
        gram = self.gram + Z.T @ (Z * weight[:, None])
        rhs = self.rhs + Z.T @ (labels * weight[:, None])
        fitted = solve(gram, rhs, assume_a='pos')
        metric = {'name': path.name, 'sha256': digest, 'frames': len(X),
                  'mse_before': float(np.mean((prediction-labels)**2)),
                  'mse_after': float(np.mean((Z@fitted-labels)**2))}
        self.gram, self.rhs = gram, rhs
        self.episodes.append(metric)
        return metric

    def save(self, path):
        path = Path(path)
        metadata = {'schema': 1, 'kind': 'episodic_external_readout',
                    'template': self.template, 'ridge': self.ridge,
                    'positive_weight': self.positive_weight, 'episodes': self.episodes}
        checksum = state_digest(metadata, self.gram, self.rhs)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, metadata=json.dumps(metadata, allow_nan=False),
                            gram=self.gram, rhs=self.rhs, sha256=checksum)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete flushed archive; link fails if the name exists.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(buffer.getvalue())
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return checksum

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            gram, rhs = data['gram'].copy(), data['rhs'].copy()
            if str(data['sha256']) != state_digest(metadata, gram, rhs):
                raise ValueError('Checkpoint checksum mismatch')
        if metadata['schema'] != 1 or metadata['kind'] != 'episodic_external_readout':
            raise ValueError('Unsupported checkpoint')
        result = cls(metadata['template'], metadata['ridge'], metadata['positive_weight'])
        if gram.shape != result.gram.shape or rhs.shape != result.rhs.shape or not np.isfinite(gram).all() or not np.isfinite(rhs).all() or not np.allclose(gram, gram.T):
            raise ValueError('Invalid checkpoint statistics')
        result.gram, result.rhs, result.episodes = gram, rhs, metadata['episodes']
        result.model()  # Also verify solvability before returning a restored learner.
        return result
