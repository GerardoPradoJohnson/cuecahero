"""Unit tests for ContinuousRateDecoder, ContinuousEmbodiment, and analog control."""
import math
import numpy as np
import pytest
from core.contracts import Action, NeuralActivity
from decoders.continuous import ContinuousRateDecoder
from embodiments.continuous import ContinuousEmbodiment
from environments.navigation_2d import Navigation2DEnvironment


def test_continuous_decoder_analog_scaling():
    # 4 groups with 10 neurons each
    groups = [list(range(i * 10, (i + 1) * 10)) for i in range(4)]
    decoder = ContinuousRateDecoder(groups=groups, tau_ms=20.0, min_rate_hz=1.0, max_rate_hz=21.0)

    # 1. Zero activity -> rates ~ 0 -> normalized to 0.0
    zero_activity = NeuralActivity(counts=np.zeros(40, dtype=np.int32), duration_ms=10.0, timestamp_ms=10.0)
    action0 = decoder.decode(zero_activity)
    assert all(math.isclose(v, 0.0, abs_tol=1e-5) for v in action0.values)

    # 2. Channel 1 fires 2 spikes per neuron in 10 ms -> 200 Hz instantaneous
    # Smoothed rate will rise smoothly
    counts = np.zeros(40, dtype=np.int32)
    counts[10:20] = 2  # group 1 fires
    act = NeuralActivity(counts=counts, duration_ms=10.0, timestamp_ms=20.0)
    action1 = decoder.decode(act)

    # Values should be analog floats in [0, 1]
    assert 0.0 < action1.values[1] <= 1.0
    assert action1.values[0] == 0.0
    assert action1.values[2] == 0.0
    assert action1.values[3] == 0.0

    # 3. Consecutive steps smooth and saturate
    for i in range(5):
        act_i = NeuralActivity(counts=counts, duration_ms=10.0, timestamp_ms=30.0 + i * 10.0)
        action_i = decoder.decode(act_i)

    assert math.isclose(action_i.values[1], 1.0, abs_tol=1e-2)

    # 4. Reset returns to 0
    decoder.reset()
    assert all(r == 0.0 for r in decoder.rates)


def test_continuous_embodiment_filtering():
    body = ContinuousEmbodiment(momentum=0.5, deadzone=0.05, limits=(0.0, 1.0))

    # Sub-deadzone input suppressed to 0
    raw1 = Action((0.02, 0.5, 0.0, 0.0))
    applied1 = body.apply(raw1)
    assert applied1.values[0] == 0.0
    assert 0.0 < applied1.values[1] < 0.5  # Filtered by momentum

    # Step again with same input -> approaches 0.5
    applied2 = body.apply(raw1)
    assert applied2.values[1] > applied1.values[1]

    # Upper limit clamped
    raw_over = Action((0.0, 1.5, 0.0, 0.0))
    applied_over = body.apply(raw_over)
    assert applied_over.values[1] <= 1.0


def test_continuous_steering_in_navigation_arena():
    env = Navigation2DEnvironment(duration=10.0)
    env.reset(seed=100)
    init_angle = env.angle

    # Gentle left steering (channel 0 = 0.3)
    env.step(Action((0.3, 0.0, 0.0, 0.0)), 0.1)
    gentle_left = env.angle
    assert gentle_left < init_angle

    # Sharp left steering (channel 0 = 1.0)
    env.reset(seed=100)
    env.step(Action((1.0, 0.0, 0.0, 0.0)), 0.1)
    sharp_left = env.angle

    # Sharp steering turns more than gentle steering
    assert abs(sharp_left - init_angle) > abs(gentle_left - init_angle)


def test_closed_loop_continuous_navigation_tick():
    env = Navigation2DEnvironment(duration=5.0)
    env.reset(42)
    body = ContinuousEmbodiment(momentum=0.2)
    groups = [[i] for i in range(4)]
    decoder = ContinuousRateDecoder(groups=groups, tau_ms=25.0)

    # Activity with graded values across channels
    counts = np.array([2, 5, 1, 0], dtype=np.int32)
    activity = NeuralActivity(counts=counts, duration_ms=10.0, timestamp_ms=10.0)

    raw_action = decoder.decode(activity)
    filtered_action = body.apply(raw_action)
    state = env.step(filtered_action, 0.033)

    assert state['agent']['speed'] >= 0.0
    assert 'time' in state
    assert len(filtered_action.values) == 4
