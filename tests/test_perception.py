"""Synthetic protocol fixtures are test-only; coverage uses the actual projection."""
import copy
import numpy as np
import pytest
from core.contracts import NeuralActivity
from core.paths import GRAPH
from decoders.calibrated import CalibratedDecoder
from sensors.contrast import ContrastVisualEncoder
from environments import CuecaHeroEnvironment


def readout_model():
    return {'schema':1,'indices':[0,1,2,3], 'mean':[0]*4,'scale':[1]*4,
            'weights':np.eye(4).tolist(),'bias':[0]*4,'tau_ms':10,
            'threshold':.45,'release':.2,'cooldown_ms':30}


def test_decoder_uses_spikes_not_absolute_time_and_no_repeated_held_action():
    model=readout_model()
    decoder=CalibratedDecoder(model)
    other=CalibratedDecoder(model)
    counts=np.array([0,0,0,1])
    assert decoder.decode(NeuralActivity(counts,10,10)).values==(0,0,0,1)
    assert other.decode(NeuralActivity(counts,10,10000)).values==(0,0,0,1)
    assert decoder.decode(NeuralActivity(counts,10,20)).values==(0,0,0,0)
    empty=CalibratedDecoder(model)
    assert empty.decode(NeuralActivity(np.zeros(4,int),10,10)).values==(0,0,0,0)


def test_decoder_rejects_corrupt_calibration():
    model=readout_model();model['weights'][0][0]=float('nan')
    with pytest.raises(ValueError):CalibratedDecoder(model)


def test_population_readout_averages_actual_spikes():
    model=readout_model()
    model['groups']=[[0,4],[1],[2],[3]]
    model['indices']=[0,1,2,3,4]
    decoder=CalibratedDecoder(model)
    assert decoder.decode(NeuralActivity(np.array([0,0,0,0,1]),10,10)).values==(1,0,0,0)
    model['indices']=[0,1,2,3]
    with pytest.raises(ValueError):CalibratedDecoder(model)


def test_neural_history_is_causal_and_resettable():
    model=readout_model();model['lags']=[0,2]
    model['mean']=[0]*8;model['scale']=[1]*8
    model['weights']=np.concatenate([np.zeros((4,4)),np.eye(4)]).tolist()
    decoder=CalibratedDecoder(model)
    spike=NeuralActivity(np.array([1,0,0,0]),10,10)
    silent=NeuralActivity(np.zeros(4),10,20)
    assert decoder.decode(spike).values==(0,0,0,0)
    assert decoder.decode(silent).values==(0,0,0,0)
    assert decoder.decode(NeuralActivity(np.zeros(4),10,30)).values==(1,0,0,0)
    decoder.reset()
    assert decoder.decode(silent).values==(0,0,0,0)


@pytest.mark.parametrize('field,value',[('tau_ms',0),('threshold',float('nan')),('release',.9),('cooldown_ms',-1)])
def test_decoder_rejects_invalid_timing(field,value):
    model=readout_model();model[field]=value
    with pytest.raises(ValueError):CalibratedDecoder(model)


def test_actual_receptor_projection_covers_all_lane_notes():
    if not GRAPH.exists():pytest.skip('Official data not installed')
    with np.load(GRAPH) as graph:
        encoder=ContrastVisualEncoder(graph['retina'],graph['uv'],graph['lamina'])
    env=CuecaHeroEnvironment();env.notes=[]
    base=encoder.encode(env.get_observation(),10).currents
    for lane in range(4):
        for ahead in [.05,.25,.55,1.,1.6,2.1]:
            env.notes=[{'id':0,'lane':lane,'at':ahead,'judgement':None}]
            encoder.reset()
            response=encoder.encode(env.get_observation(),10).currents
            assert np.max(np.abs(response-base))>1, (lane,ahead)
