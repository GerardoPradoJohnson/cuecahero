import copy
import json
from pathlib import Path

import numpy as np
import pytest

from core.contracts import Action, NeuralActivity
from decoders.calibrated import CalibratedDecoder
from training.rl import ReinforcementReadoutTrainer


def sample_model():
    """Create a minimal valid model template for testing."""
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    lags = [0, 1]
    dim = len(groups) * len(lags)
    return {
        'schema': 1,
        'groups': groups,
        'indices': [0, 1, 2, 3, 4, 5, 6, 7],
        'lags': lags,
        'mean': [0.0] * dim,
        'scale': [1.0] * dim,
        'weights': np.zeros((dim, 4), dtype=float).tolist(),
        'bias': [0.0, 0.0, 0.0, 0.0],
        'threshold': 0.5,
        'release': 0.2,
        'tau_ms': 10.0,
        'cooldown_ms': 30.0,
        'sensor': {'type': 'contrast_visual'},
        'graph_sha256': 'dummy_sha',
        'source_ids': ['0', '1', '2', '3', '4', '5', '6', '7']
    }


def test_trainer_initialization():
    model = sample_model()
    trainer = ReinforcementReadoutTrainer(model, lr=0.01, gamma=0.9)
    assert trainer.dim == 8
    assert trainer.weights.shape == (8, 4)
    assert trainer.bias.shape == (4,)
    assert trainer.opt_step == 0
    assert len(trainer.episodes) == 0


def test_step_exploration_vs_deterministic():
    model = sample_model()
    trainer = ReinforcementReadoutTrainer(model)
    norm_x = np.ones(8, dtype=float)
    
    # Deterministic step: since weights and bias are 0 and threshold is 0.5, rate is 0 < threshold -> no press
    action_det, rec_det = trainer.step(norm_x, current_time_ms=100.0, explore=False)
    assert action_det.values == (0.0, 0.0, 0.0, 0.0)
    
    # Stochastic step with RNG: probability should be valid in [0, 1]
    rng = np.random.default_rng(42)
    action_stoch, rec_stoch = trainer.step(norm_x, current_time_ms=200.0, explore=True, rng=rng)
    assert len(action_stoch.values) == 4
    assert np.all((rec_stoch['probs'] >= 0.0) & (rec_stoch['probs'] <= 1.0))


def test_policy_gradient_update():
    model = sample_model()
    trainer = ReinforcementReadoutTrainer(model, lr=0.1, gamma=0.95)
    
    # Create synthetic trajectory of 10 steps
    trajectory = []
    rewards = np.zeros((10, 4), dtype=float)
    rng = np.random.default_rng(123)
    
    for t in range(10):
        norm_x = np.ones(8, dtype=float) * (t + 1)
        action, rec = trainer.step(norm_x, current_time_ms=t * 33.3, explore=True, rng=rng)
        trajectory.append(rec)
        
    # Give positive reward on lane 0 at step 5
    rewards[5, 0] = 1.0
    # Give negative reward on lane 1 at step 7
    rewards[7, 1] = -1.0
    
    initial_weights = trainer.weights.copy()
    initial_bias = trainer.bias.copy()
    
    grad_norm = trainer.update_policy(trajectory, rewards)
    assert grad_norm > 0.0
    assert trainer.opt_step == 1
    
    # Weights and bias should have changed
    assert not np.array_equal(trainer.weights, initial_weights)
    assert not np.array_equal(trainer.bias, initial_bias)


def test_checkpoint_roundtrip_and_corruption(tmp_path):
    model = sample_model()
    trainer = ReinforcementReadoutTrainer(model, lr=0.005)
    
    # Do one update
    norm_x = np.ones(8, dtype=float)
    _, rec = trainer.step(norm_x, 100.0, explore=True)
    trainer.update_policy([rec], np.array([[1.0, 0.0, 0.0, 0.0]]))
    trainer.episodes.append({'name': 'episode-0001', 'seed': 42, 'score': 100, 'sha256': 'temp'})
    
    ckpt_path = tmp_path / 'episode-0001.npz'
    digest = trainer.save(ckpt_path)
    trainer.episodes[0]['sha256'] = digest
    
    # Restore trainer from checkpoint
    restored = ReinforcementReadoutTrainer.load(ckpt_path)
    np.testing.assert_array_equal(restored.weights, trainer.weights)
    np.testing.assert_array_equal(restored.bias, trainer.bias)
    np.testing.assert_array_equal(restored.m, trainer.m)
    np.testing.assert_array_equal(restored.v, trainer.v)
    assert restored.opt_step == trainer.opt_step
    assert restored.baseline_count == trainer.baseline_count
    assert len(restored.episodes) == 1
    assert restored.episodes[0]['name'] == 'episode-0001'
    
    # Export model and verify CalibratedDecoder loads it
    exported_model = restored.model()
    decoder = CalibratedDecoder(exported_model)
    assert decoder.feature_count == 4
    
    # Test corruption detection
    with np.load(ckpt_path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays['weights'] = arrays['weights'] + 0.1
    corrupt_path = tmp_path / 'corrupt.npz'
    np.savez_compressed(corrupt_path, **arrays)
    
    with pytest.raises(ValueError, match='checksum mismatch'):
        ReinforcementReadoutTrainer.load(corrupt_path)
