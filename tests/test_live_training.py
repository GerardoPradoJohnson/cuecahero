"""Verification for online continuous generational reinforcement learning."""
import json
import numpy as np
from core.paths import ROOT
from experiments.runtime import Experiment


def test_live_training_starts_from_scratch(tmp_path):
    config = json.loads((ROOT / "experiments/cueca_hero.json").read_text())
    config['environment']['bars'] = 1  # Short episode for fast test
    experiment = Experiment(config, {'total_neurons': 0, 'indices': []}, sessions_dir=tmp_path)
    
    assert experiment.live_training_enabled is True
    assert experiment.generation == 0
    assert experiment.training_history == []
    
    # Load brain to initialize trainer
    experiment.load_brain()
    assert experiment.trainer is not None
    
    # Verify weights start zeroed out ("sabiendo nada")
    assert np.all(experiment.trainer.weights == 0)
    assert np.all(experiment.trainer.bias == -0.6)
    
    experiment.close()


def test_live_training_advances_generation_without_stopping(tmp_path):
    config = json.loads((ROOT / "experiments/cueca_hero.json").read_text())
    config['environment']['bars'] = 1  # 4 notes total
    experiment = Experiment(config, {'total_neurons': 0, 'indices': []}, sessions_dir=tmp_path)
    experiment.load_brain()
    experiment.driver = 'neural'
    experiment.mode = 'running'
    
    # Run until one episode finishes
    max_ticks = 200
    ticks = 0
    while experiment.generation == 0 and ticks < max_ticks:
        experiment.tick()
        ticks += 1
        
    # Episode finished: should have advanced to Generation 1 automatically!
    assert experiment.generation >= 1
    assert len(experiment.training_history) >= 1
    gen0 = experiment.training_history[0]
    assert gen0['generation'] == 0
    assert 'accuracy' in gen0
    assert 'score' in gen0
    
    # The mode must still be 'running' (does not stop / pause!)
    assert experiment.mode == 'running'
    
    # The weights must have updated via policy gradient
    assert not np.all(experiment.trainer.weights == 0)
    
    # Snapshot must reflect live training metadata
    snapshot = experiment.snapshot
    assert 'live_training' in snapshot
    assert snapshot['live_training']['generation'] >= 1
    assert len(snapshot['live_training']['history']) >= 1
    assert snapshot['live_training']['is_training'] is True
    
    # Reset training command resets back to Gen 0
    experiment.control({'type': 'reset_training'})
    assert experiment.generation == 0
    assert len(experiment.training_history) == 0
    assert np.all(experiment.trainer.weights == 0)
    
    experiment.close()
