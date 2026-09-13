import json
from pathlib import Path

import numpy as np
import pytest

from core.contracts import RewardSignal
from core.malecns import MaleCNSCore
from decoders.calibrated import CalibratedDecoder
from plasticity.reward_modulated import RewardModulatedPlasticity
from sensors.contrast import ContrastVisualEncoder


def test_plasticity_binds_to_real_malecns_graph():
    brain = MaleCNSCore(backend='cpu')
    model = json.loads(Path('config/perception-readout.json').read_text())
    encoder = ContrastVisualEncoder(
        brain.state.retina, brain.state.uv, brain.state.lamina,
        **model['sensor']
    )
    decoder = CalibratedDecoder(model)
    
    p = RewardModulatedPlasticity(eta=0.002, minimum_fraction=0.95, maximum_fraction=1.05)
    p.bind(brain, encoder.retina, decoder.indices, 'test-graph-hash')
    
    assert p.bound is True
    # The active edge count follows the configured visual viewport.
    edge_count = len(p.edges)
    assert edge_count > 0
    assert len(p.pre) == edge_count
    assert len(p.post) == edge_count
    assert len(p.baseline) == edge_count
    assert np.all(p.baseline != 0)
    assert np.all(np.isfinite(p.baseline))


def test_selective_synaptic_modification_and_erase():
    brain = MaleCNSCore(backend='cpu')
    model = json.loads(Path('config/perception-readout.json').read_text())
    encoder = ContrastVisualEncoder(
        brain.state.retina, brain.state.uv, brain.state.lamina,
        **model['sensor']
    )
    decoder = CalibratedDecoder(model)
    
    p = RewardModulatedPlasticity(eta=0.01, minimum_fraction=0.95, maximum_fraction=1.05)
    p.bind(brain, encoder.retina, decoder.indices, 'test-graph-hash')
    
    initial_weights = brain.state.weight[p.edges].copy()
    assert np.array_equal(initial_weights, p.baseline)
    
    # 1. When all counts are 0, reward does not modify any weights
    brain.state.counts[:] = 0
    p.apply(brain, RewardSignal(1.0, 'cueca_hero', 0.0))
    np.testing.assert_array_equal(brain.state.weight[p.edges], p.baseline)
    assert p.status(brain)['changed_edges'] == 0
    
    # 2. Selectively activate only ONE specific pre/post pair
    target_edge = 42
    pre_neuron = p.pre[target_edge]
    post_neuron = p.post[target_edge]
    
    brain.state.counts[pre_neuron] = 5
    brain.state.counts[post_neuron] = 5
    
    # Apply positive reward
    p.apply(brain, RewardSignal(1.0, 'cueca_hero', 0.1))
    
    status = p.status(brain)
    assert status['changed_edges'] > 0
    
    # Verify target edge was modified
    w_target = brain.state.weight[p.edges[target_edge]]
    base_target = p.baseline[target_edge]
    assert w_target != base_target
    
    # Verify edge weights remain within bounds
    fraction = brain.state.weight[p.edges] / p.baseline
    assert np.all(fraction >= 0.95 - 1e-6)
    assert np.all(fraction <= 1.05 + 1e-6)
    
    # 3. Verify erase() restores exact baseline bit for bit
    p.erase(brain)
    np.testing.assert_array_equal(brain.state.weight[p.edges], p.baseline)
    assert p.status(brain)['changed_edges'] == 0
    assert p.updates == 0


def test_plasticity_checkpoint_with_real_brain(tmp_path):
    brain = MaleCNSCore(backend='cpu')
    model = json.loads(Path('config/perception-readout.json').read_text())
    encoder = ContrastVisualEncoder(
        brain.state.retina, brain.state.uv, brain.state.lamina,
        **model['sensor']
    )
    decoder = CalibratedDecoder(model)
    
    p = RewardModulatedPlasticity(eta=0.01, minimum_fraction=0.95, maximum_fraction=1.05)
    p.bind(brain, encoder.retina, decoder.indices, 'test-graph-hash')
    
    # Trigger active coincidence on first 10 edges
    for e in range(10):
        brain.state.counts[p.pre[e]] = 2
        brain.state.counts[p.post[e]] = 2
        
    p.apply(brain, RewardSignal(1.0, 'cueca_hero', 0.1))
    modified_weights = brain.state.weight[p.edges].copy()
    
    ckpt_file = tmp_path / 'plastic_memory.npz'
    digest = p.checkpoint(brain, ckpt_file)
    assert ckpt_file.exists()
    assert isinstance(digest, str) and len(digest) == 64
    
    # Erase brain
    p.erase(brain)
    np.testing.assert_array_equal(brain.state.weight[p.edges], p.baseline)
    
    # Restore from checkpoint
    p.restore(brain, ckpt_file)
    np.testing.assert_array_equal(brain.state.weight[p.edges], modified_weights)
    assert p.status(brain)['changed_edges'] > 0
