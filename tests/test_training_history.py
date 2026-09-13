"""Unit tests for training history and checkpoint inspection API."""
import pytest
from training.history import list_training_runs, get_run_curve


def test_list_training_runs():
    runs = list_training_runs()
    assert len(runs) > 0
    run_ids = [r['id'] for r in runs]
    assert 'rl/rl-v3' in run_ids
    assert 'plasticity/run1' in run_ids

    rl_v3 = next(r for r in runs if r['id'] == 'rl/rl-v3')
    assert rl_v3['type'] == 'rl_generational'
    assert rl_v3['checkpoints_count'] >= 24
    assert 'accuracy_gain' in rl_v3['summary']


def test_get_run_curve_rl_v3():
    curve_data = get_run_curve('rl/rl-v3')
    assert curve_data['kind'] == 'generation_learning_curve_report'
    curve = curve_data['curve']
    assert len(curve) >= 24

    # Checkpoint metadata
    first = curve[0]
    last = curve[-1]
    assert first['checkpoint'] == 'baseline'
    assert 'mean_accuracy' in first
    assert 'mean_score' in first

    # Verify gain
    assert last['mean_accuracy'] > first['mean_accuracy']
    assert last['mean_score'] > first['mean_score']
    assert last['mean_wrong'] < first['mean_wrong']


def test_get_run_curve_plasticity():
    curve_data = get_run_curve('plasticity/run1')
    assert 'curve' in curve_data
    curve = curve_data['curve']
    assert len(curve) >= 5
    assert curve[0]['checkpoint'] == 'episode-0001.npz'


def test_get_nonexistent_run_raises():
    with pytest.raises(FileNotFoundError):
        get_run_curve('nonexistent/run_xyz')
