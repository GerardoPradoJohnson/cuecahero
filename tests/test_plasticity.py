import copy
from types import SimpleNamespace

import numpy as np
import pytest

from core.contracts import RewardSignal
from plasticity.reward_modulated import RewardModulatedPlasticity


def brain():
    # Existing edges: 0->2, 0->3, 1->2. Only edges ending in readout 2 are selected.
    state=SimpleNamespace(n=4, ptr=np.array([0,2,3,3,3]),
        post=np.array([2,3,2],np.int32), weight=np.array([2.,-1.,-3.],np.float32),
        counts=np.zeros(4,np.int32))
    return SimpleNamespace(state=state)


def model():
    b=brain();p=RewardModulatedPlasticity(eta=.01,minimum_fraction=.9,maximum_fraction=1.1)
    p.bind(b,[0,1],[2],'graph');return b,p


def test_reward_changes_only_identified_existing_edges_and_respects_sign_bounds():
    b,p=model();original=b.state.weight.copy();b.state.counts[:]=[1,1,1,1]
    p.apply(b,RewardSignal(1,'cueca_hero',0))
    assert b.state.weight[0]>original[0] and b.state.weight[2]<original[2]
    assert b.state.weight[1]==original[1]
    for _ in range(100):p.apply(b,RewardSignal(1,'cueca_hero',0))
    np.testing.assert_allclose(b.state.weight[p.edges]/p.baseline,1.1)
    for _ in range(300):p.apply(b,RewardSignal(-1,'cueca_hero',0))
    np.testing.assert_allclose(b.state.weight[p.edges]/p.baseline,.9)


def test_no_coactivity_means_no_change_and_erase_clears_memory():
    b,p=model();p.apply(b,RewardSignal(1,'cueca_hero',0))
    np.testing.assert_array_equal(b.state.weight[p.edges],p.baseline)
    b.state.counts[:]=1;p.apply(b,RewardSignal(1,'cueca_hero',0));p.erase(b)
    np.testing.assert_array_equal(b.state.weight[p.edges],p.baseline)
    assert p.status(b)['changed_edges']==0 and p.updates==0


def test_checkpoint_roundtrip_and_corruption(tmp_path):
    b,p=model();b.state.counts[:]=1;p.apply(b,RewardSignal(.6,'cueca_hero',0))
    path=tmp_path/'memory.npz';p.checkpoint(b,path);expected=b.state.weight.copy()
    restored_brain,restored=model();restored.restore(restored_brain,path)
    np.testing.assert_array_equal(restored_brain.state.weight,expected)
    np.testing.assert_array_equal(restored.eligibility,p.eligibility)
    with pytest.raises(FileExistsError):p.checkpoint(b,path)
    with np.load(path) as data: arrays={name:data[name] for name in data.files}
    arrays['weights']=arrays['weights'].copy();arrays['weights'][0]*=.99
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='checksum'):restored.restore(restored_brain,path)


def test_invalid_reward_rejected():
    b,p=model()
    with pytest.raises(ValueError):p.apply(b,RewardSignal(float('nan'),'cueca_hero',0))
    with pytest.raises(ValueError):p.apply(b,RewardSignal(1,'other',0))
