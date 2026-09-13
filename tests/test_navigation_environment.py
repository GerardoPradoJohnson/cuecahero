"""Comprehensive tests for Navigation2DEnvironment and decoupling with MaleCNSCore."""
import math
import numpy as np
import pytest
from core.contracts import Action, Observation, RewardSignal, Environment
from environments.navigation_2d import (
    Navigation2DEnvironment,
    ARENA_WIDTH,
    ARENA_HEIGHT,
    ARENA_MARGIN,
)


def test_navigation_protocol_compliance():
    env = Navigation2DEnvironment(duration=10.0)
    assert isinstance(env, Environment)
    state = env.reset(seed=42)
    assert isinstance(state, dict)
    assert state['environment'] == 'navigation_2d'
    assert state['time'] == 0.0

    obs = env.get_observation()
    assert isinstance(obs, Observation)
    assert isinstance(obs.rgb, np.ndarray)
    assert obs.rgb.shape == (ARENA_HEIGHT, ARENA_WIDTH, 3)
    assert obs.rgb.dtype == np.uint8
    assert math.isclose(obs.timestamp, 0.0)

    reward = env.get_reward()
    assert isinstance(reward, RewardSignal)
    assert reward.source == 'navigation_2d'

    assert not env.is_done()


def test_seed_determinism():
    env1 = Navigation2DEnvironment(duration=15.0)
    env2 = Navigation2DEnvironment(duration=15.0)
    env1.reset(123)
    env2.reset(123)

    assert env1.x == env2.x and env1.y == env2.y
    assert env1.angle == env2.angle
    assert env1.target == env2.target
    assert len(env1.obstacles) == len(env2.obstacles)
    for ob1, ob2 in zip(env1.obstacles, env2.obstacles):
        assert ob1 == ob2

    # Stepping with same actions produces identical results
    actions = [
        Action((1.0, 1.0, 0.0, 0.0)),
        Action((0.0, 1.0, 0.0, 0.0)),
        Action((0.0, 0.0, 1.0, 0.0)),
    ]
    for act in actions:
        s1 = env1.step(act, 0.05)
        s2 = env2.step(act, 0.05)
        assert s1['agent'] == s2['agent']
        np.testing.assert_array_equal(env1.get_observation().rgb, env2.get_observation().rgb)


def test_turning_and_acceleration():
    env = Navigation2DEnvironment(duration=10.0)
    env.reset(42)
    initial_angle = env.angle
    initial_speed = env.speed

    # Turn left (channel 0)
    env.step(Action((1.0, 0.0, 0.0, 0.0)), 0.1)
    assert env.angle < initial_angle  # Left turn decreases angle

    # Turn right (channel 2)
    mid_angle = env.angle
    env.step(Action((0.0, 0.0, 1.0, 0.0)), 0.2)
    assert env.angle > mid_angle      # Right turn increases angle

    # Accelerate forward (channel 1)
    env.step(Action((0.0, 1.0, 0.0, 0.0)), 0.1)
    assert env.speed > initial_speed

    # Brake (channel 3)
    fast_speed = env.speed
    env.step(Action((0.0, 0.0, 0.0, 1.0)), 0.1)
    assert env.speed < fast_speed


def test_arena_boundary_constraint():
    env = Navigation2DEnvironment(duration=10.0)
    env.reset(42)
    # Move towards left wall
    env.x = ARENA_MARGIN + env.agent_radius + 1.0
    env.angle = math.pi  # Heading directly left
    env.speed = 60.0

    env.step(Action((0.0, 1.0, 0.0, 0.0)), 0.1)
    assert env.x >= ARENA_MARGIN + env.agent_radius
    assert env.wrong > 0
    assert env.get_reward().magnitude < 0


def test_target_pickup_and_reward():
    env = Navigation2DEnvironment(duration=10.0)
    env.reset(42)
    # Manually place target right in front of the agent
    env.target['x'] = env.x
    env.target['y'] = env.y - 12.0
    env.angle = -math.pi / 2.0  # Facing straight up towards target
    env.last_distance = env._distance_to_target()

    # Step forward to capture
    env.step(Action((0.0, 1.0, 0.0, 0.0)), 0.1)
    assert env.hits == 1
    assert env.score >= 100
    assert env.combo == 1
    assert env.get_reward().magnitude >= 1.0


def test_obstacle_collision():
    env = Navigation2DEnvironment(duration=10.0)
    env.reset(42)
    # Place an obstacle directly in front of agent
    env.obstacles[0]['x'] = env.x
    env.obstacles[0]['y'] = env.y - 15.0
    env.obstacles[0]['r'] = 14.0
    env.angle = -math.pi / 2.0  # Facing towards obstacle

    env.step(Action((0.0, 1.0, 0.0, 0.0)), 0.1)
    assert env.misses == 1
    assert env.combo == 0
    assert env.get_reward().magnitude < 0


def test_episode_finishes_at_duration():
    env = Navigation2DEnvironment(duration=1.0)
    env.reset(42)
    steps = int(1.0 / 0.05) + 1
    for _ in range(steps):
        env.step(Action(), 0.05)
    assert env.is_done()


def test_decoupling_with_malecns_core():
    """Verify that MaleCNSCore connects to Navigation2DEnvironment without altering MaleCNSCore."""
    from core.malecns import MaleCNSCore
    from sensors.visual import VisualEncoder
    from decoders.directional import DirectionalDecoder
    import json
    from core.paths import DATA

    # Load brain
    brain = MaleCNSCore(backend='cpu')
    assert brain.n == 166700

    # Load visual encoder
    encoder = VisualEncoder(retina=brain.state.retina, uv=brain.state.uv, lamina=brain.state.lamina)

    # Load directional decoder using actual descending neurons
    manifest = json.loads((DATA / 'outputs/doom/malecns_v1/manifest.json').read_text())['readouts']
    specs = [
        {'type': 'DNa02', 'side': 'L'},
        {'type': 'DNpe017', 'side': 'L'},
        {'type': 'DNpe017', 'side': 'R'},
        {'type': 'DNa02', 'side': 'R'},
    ]
    groups = [[r['index'] for r in manifest if r['type'] == s['type'] and r['side'] == s['side']] for s in specs]
    decoder = DirectionalDecoder(groups=groups, threshold_hz=0.5, cooldown_ms=40.0)

    # Instantiate navigation environment
    env = Navigation2DEnvironment(duration=2.0)
    env.reset(42)

    # Step simulation loop
    dt = 0.03333333333333333
    neural_step_ms = 10.0
    steps_run = 0

    for _ in range(5):
        obs = env.get_observation()
        stim = encoder.encode(obs, neural_step_ms)
        activity = brain.advance(stim, neural_step_ms)
        action = decoder.decode(activity)
        state = env.step(action, dt)
        assert 'agent' in state
        assert 'score' in state
        assert activity.counts.sum() >= 0
        steps_run += 1

    assert steps_run == 5
    assert env.time > 0.0
