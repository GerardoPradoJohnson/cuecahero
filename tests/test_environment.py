"""Gameplay assertions use deterministic charts, never simulated neural data."""
import numpy as np
import pytest
from core.contracts import Action
from environments import CuecaHeroEnvironment
from core.clock import SimulationClock
from decoders import DirectionalDecoder
from core.contracts import NeuralActivity


def advance_to(env, t):
    while env.time < t - 1e-10:
        env.step(Action(), min(.1, t-env.time))


def test_chart_reproducible_and_reset_clears_all_score():
    env = CuecaHeroEnvironment()
    env.reset(17)
    chart = [dict(n) for n in env.notes]
    advance_to(env, 4)
    assert env.misses > 0
    env.reset(17)
    assert env.notes == chart and env.score == env.combo == env.misses == 0


def test_timed_hit_only_same_lane_and_no_double_credit():
    env = CuecaHeroEnvironment()
    note = env.notes[0]
    advance_to(env, note['at'] - .02)
    buttons = tuple(float(i==note['lane']) for i in range(4))
    env.step(Action(buttons), .02)
    assert env.hits == 1 and env.score == 100 and env.get_reward().magnitude == 1
    env.step(Action(buttons), .01)
    assert env.hits == 1 and env.score == 100 and env.wrong == 0


def test_silent_session_expires_every_note_and_finishes():
    env = CuecaHeroEnvironment(bars=1)
    while not env.is_done():
        env.step(Action(), 1/30)
    assert env.misses == len(env.notes) and env.hits == 0
    score = env.score
    env.step(Action((1,1,1,1)), .1)
    assert env.score == score and env.get_reward().magnitude == 0


def test_observation_does_not_expose_hidden_future_chart():
    env = CuecaHeroEnvironment()
    before = env.get_observation().rgb.copy()
    env.notes[-1]['lane'] = (env.notes[-1]['lane'] + 1) % 4
    np.testing.assert_array_equal(env.get_observation().rgb, before)
    assert vars(env.get_observation()).keys() == {'rgb','timestamp'}


def test_clock_keeps_neural_and_game_time_separate():
    clock = SimulationClock(game_step=.05, neural_step_ms=10)
    clock.advance(False)
    clock.advance(True)
    assert clock.game_seconds == .1 and clock.neural_ms == 10
    with pytest.raises(ValueError):
        SimulationClock(neural_step_ms=.15)


def test_decoder_needs_activity_and_respects_cooldown():
    decoder = DirectionalDecoder([[0],[1],[2],[3]], cooldown_ms=100)
    silent = NeuralActivity(np.zeros(4, np.int32), 10, 10)
    assert decoder.decode(silent).values == (0,0,0,0)
    counts = np.array([0,0,1,0], np.int32)
    assert decoder.decode(NeuralActivity(counts, 10, 20)).values == (0,0,1,0)
    assert decoder.decode(NeuralActivity(counts, 10, 30)).values == (0,0,0,0)
    assert decoder.decode(NeuralActivity(counts, 10, 120)).values == (0,0,1,0)


def test_lane_feedback_survives_skipped_frames_and_reset():
    env=CuecaHeroEnvironment()
    note=env.notes[0];lane=note['lane']
    advance_to(env,note['at']-.02)
    env.step(Action(tuple(float(i==lane) for i in range(4))),.02)
    captured=env.get_state()
    env.step(Action(),.03)
    assert env.get_state()['events']==[]
    assert env.get_state()['lanes'][lane]['hits']==1
    assert env.get_state()['lanes'][lane]['last_event']['kind']=='perfect'
    env.step(Action(tuple(float(i==lane) for i in range(4))),.01)
    assert captured['lanes'][lane]['wrong']==0
    assert env.get_state()['lanes'][lane]['wrong']==1
    env.reset()
    assert all(l['last_event'] is None and l['hits']==l['misses']==l['wrong']==0 for l in env.get_state()['lanes'])


def test_sustain_note_hold_and_release():
    env = CuecaHeroEnvironment()
    # Inject a sustain note on lane 1
    env.notes = [
        {'id': 0, 'lane': 1, 'at': 1.0, 'sustain': 0.5, 'judgement': None}
    ]
    env.duration = 3.0
    advance_to(env, 0.98)
    
    # Hit note head
    action_down = Action((0.0, 1.0, 0.0, 0.0))
    env.step(action_down, 0.02)
    assert env.hits == 1 and env.combo == 1
    assert 1 in env.active_holds
    base_score = env.score
    assert base_score == 100
    
    # Continue holding for 0.2s: hold points accumulate
    env.step(action_down, 0.1)
    env.step(action_down, 0.1)
    assert env.score > base_score
    assert 1 in env.active_holds
    
    # Releasing stops active hold cleanly without wrong/miss
    action_up = Action((0.0, 0.0, 0.0, 0.0))
    env.step(action_up, 0.05)
    assert 1 not in env.active_holds
    assert env.wrong == 0 and env.misses == 0


def test_held_key_does_not_hit_following_note_without_release():
    env = CuecaHeroEnvironment()
    env.notes = [{'id': i, 'lane': 0, 'at': t, 'judgement': None}
                 for i, t in enumerate((1., 1.4))]
    advance_to(env, .98)
    down = Action((1, 0, 0, 0))
    env.step(down, .02)
    for _ in range(5):
        env.step(down, .1)
    assert env.hits == 1 and env.wrong == 0
    env.step(Action(), .01)
    env.step(down, .01)
    assert env.hits == 2


def test_sustain_completion_does_not_add_a_second_hit_or_wrong_press():
    env = CuecaHeroEnvironment()
    env.notes = [{'id': 0, 'lane': 0, 'at': 1., 'sustain': .5, 'judgement': None}]
    advance_to(env, .98)
    down = Action((1, 0, 0, 0))
    env.step(down, .02)
    for _ in range(7):
        env.step(down, .1)
    assert not env.active_holds
    assert env.hits == env.lanes[0]['hits'] == 1
    assert env.wrong == env.lanes[0]['wrong'] == 0

