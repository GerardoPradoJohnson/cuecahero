"""Unit tests for training history without relying on ignored local runs."""
import json
import pytest
import training.history as history


@pytest.fixture(autouse=True)
def training_runs(tmp_path, monkeypatch):
    training = tmp_path / 'training'
    plasticity = tmp_path / 'plasticity'
    monkeypatch.setattr(history, 'ROOT', tmp_path)
    monkeypatch.setattr(history, 'TRAINING_DIR', training)
    monkeypatch.setattr(history, 'PLASTICITY_DIR', plasticity)

    rl = training / 'rl-v3'
    rl.mkdir(parents=True)
    curve = []
    for i in range(24):
        name = 'baseline' if i == 0 else f'episode-{i:04d}.npz'
        curve.append(dict(generation=i, checkpoint=name, mean_accuracy=20+i,
                          mean_score=100*i, mean_wrong=50-i))
        if name != 'baseline':
            (rl / name).write_bytes(b'checkpoint')
    (rl / 'generation_curve.json').write_text(json.dumps(dict(
        kind='generation_learning_curve_report', curve=curve,
        summary=dict(accuracy_gain=23))))

    plastic = plasticity / 'run1'
    plastic.mkdir(parents=True)
    episodes = [dict(index=i+1, name=f'episode-{i+1:04d}', score=i) for i in range(5)]
    (plastic / 'progress.json').write_text(json.dumps(episodes))


def test_list_training_runs():
    runs = history.list_training_runs()
    run_ids = [r['id'] for r in runs]
    assert 'rl/rl-v3' in run_ids
    assert 'plasticity/run1' in run_ids
    rl_v3 = next(r for r in runs if r['id'] == 'rl/rl-v3')
    assert rl_v3['type'] == 'rl_generational'
    assert rl_v3['checkpoints_count'] == 23
    assert 'accuracy_gain' in rl_v3['summary']


def test_get_run_curve_rl_v3():
    curve_data = history.get_run_curve('rl/rl-v3')
    assert curve_data['kind'] == 'generation_learning_curve_report'
    curve = curve_data['curve']
    assert len(curve) == 24
    assert curve[0]['checkpoint'] == 'baseline'
    assert curve[-1]['mean_accuracy'] > curve[0]['mean_accuracy']
    assert curve[-1]['mean_score'] > curve[0]['mean_score']
    assert curve[-1]['mean_wrong'] < curve[0]['mean_wrong']


def test_get_run_curve_plasticity():
    curve = history.get_run_curve('plasticity/run1')['curve']
    assert len(curve) == 5
    assert curve[0]['checkpoint'] == 'episode-0001.npz'


def test_get_nonexistent_run_raises():
    with pytest.raises(FileNotFoundError):
        history.get_run_curve('nonexistent/run_xyz')
