"""Unit tests for multimodal sensory encoders (auditory, proprioception, composite)."""
import numpy as np
import pytest
from core.contracts import Observation, MultimodalObservation, NeuralStimulus
from sensors.auditory import AuditoryEncoder
from sensors.proprioception import ProprioceptiveEncoder
from sensors.multimodal import CompositeMultimodalEncoder


def test_auditory_encoder():
    targets = [100, 101, 102]
    encoder = AuditoryEncoder(target_indices=targets, baseline_current=5.0, gain=30.0)

    # 1. Silent observation -> baseline current
    silent_obs = MultimodalObservation(rgb=np.zeros((10, 10, 3), dtype=np.uint8), timestamp=0.0, audio=None)
    stim0 = encoder.encode(silent_obs, duration_ms=10.0)
    assert len(stim0.indices) == 3
    assert np.allclose(stim0.currents, 5.0, atol=1e-3)

    # 2. Loud sound wave -> current increases above baseline
    loud_wave = np.sin(np.linspace(0, 10, 100)) * 0.8
    loud_obs = MultimodalObservation(rgb=np.zeros((10, 10, 3), dtype=np.uint8), timestamp=0.01, audio=loud_wave)
    stim1 = encoder.encode(loud_obs, duration_ms=10.0)
    assert stim1.currents[0] > 5.5

    # 3. Reset clears integrated energy
    encoder.reset()
    assert encoder.filtered_energy == 0.0


def test_proprioceptive_encoder():
    targets = [200, 201]
    encoder = ProprioceptiveEncoder(target_indices=targets, tonic_bias=6.0, velocity_gain=0.5, impact_gain=20.0)

    # 1. Stationary fly
    obs_idle = MultimodalObservation(
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp=0.0,
        proprioception={'speed': 0.0, 'angular_velocity': 0.0, 'collision': 0.0},
    )
    stim_idle = encoder.encode(obs_idle, duration_ms=10.0)
    assert np.allclose(stim_idle.currents, 6.0, atol=0.1)

    # 2. Forward flight speed increases tonic current
    obs_flying = MultimodalObservation(
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp=0.05,
        proprioception={'speed': 40.0, 'angular_velocity': 0.0, 'collision': 0.0},
    )
    stim_flying = encoder.encode(obs_flying, duration_ms=10.0)
    assert stim_flying.currents[0] > stim_idle.currents[0]

    # 3. Collision produces large phasic spike
    obs_collision = MultimodalObservation(
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp=0.06,
        proprioception={'speed': 10.0, 'angular_velocity': 2.0, 'collision': 1.0},
    )
    stim_collision = encoder.encode(obs_collision, duration_ms=10.0)
    assert stim_collision.currents[0] > stim_flying.currents[0]


def test_composite_multimodal_fusion():
    auditory = AuditoryEncoder(target_indices=[10, 11])
    proprio = ProprioceptiveEncoder(target_indices=[20, 21, 22])
    composite = CompositeMultimodalEncoder([auditory, proprio])

    obs = MultimodalObservation(
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp=0.0,
        audio=np.ones(10, dtype=np.float32) * 0.5,
        proprioception={'speed': 25.0, 'angular_velocity': 0.5, 'collision': 0.0},
    )

    stim = composite.encode(obs, duration_ms=10.0)
    # Total indices must be 2 + 3 = 5
    assert len(stim.indices) == 5
    assert len(stim.currents) == 5
    np.testing.assert_array_equal(stim.indices, [10, 11, 20, 21, 22])
    assert all(c > 0.0 for c in stim.currents)
