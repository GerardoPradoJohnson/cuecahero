"""Explicit synthetic numerical fixture. Never available to application runtime."""
import numpy as np
import pytest
from core.malecns import MaleCNSCore
from core.contracts import NeuralStimulus

@pytest.fixture
def mock_graph(tmp_path):
    n=4
    path=tmp_path/'mock-test-only.npz'
    np.savez(path, ptr=np.array([0,3,5,6,7],np.int64), post=np.array([0,1,2,2,3,1,0],np.int32),
             weight=np.array([100,35,-20,45,-15,20,-10],np.float32), ids=np.arange(n,dtype=np.int64),
             retina=np.array([],np.int32), uv=np.empty((0,2),np.float32), lamina=np.array([],np.int32),
             sugar=np.array([],np.int32), superclass=np.array(['test']*n))
    return path


@pytest.mark.parametrize('backend', ['cpu', 'native'])
def test_arbitrary_current_adapter_matches_brian2_oracle(mock_graph, backend):
    from core.paths import ROOT
    import platform
    library = ROOT / 'build' / ('neural.dll' if platform.system() == 'Windows' else 'libneural.dylib')
    if backend == 'native' and not library.exists():
        pytest.skip('Optional native compiler/library unavailable')
    import brian2 as b
    b.start_scope()
    b.prefs.codegen.target='numpy'
    b.defaultclock.dt=.1*b.ms
    neurons=b.NeuronGroup(4, '''
    dv/dt = (-52*mV-v+drive+g)/(20*ms) : volt (unless refractory)
    dg/dt = -g/(5*ms) : volt (unless refractory)
    drive : volt
    ''', method='exact', threshold='v > -45*mV', reset='v=-52*mV;g=0*mV', refractory=2.2*b.ms)
    neurons.v=-52*b.mV
    synapses=b.Synapses(neurons,neurons,'w : volt',on_pre='g += w',delay=1.8*b.ms)
    synapses.connect(i=[0,0,0,1,1,2,3],j=[0,1,2,2,3,1,0])
    synapses.w=np.array([100,35,-20,45,-15,20,-10])*b.mV
    spike=b.SpikeMonitor(neurons)
    network=b.Network(neurons,synapses,spike)
    brains=[MaleCNSCore(mock_graph,backend=backend)]
    previous=np.zeros(4,int)
    for currents in ([12,0,0,12],[0,0,0,0],[18,0,0,18],[0,14,0,0]):
        neurons.drive=np.array(currents)*b.mV
        network.run(40*b.ms)
        expected=np.asarray(spike.count)-previous
        previous=np.asarray(spike.count).copy()
        for brain in brains:
            activity=brain.advance(NeuralStimulus(np.arange(4),np.array(currents)),40)
            np.testing.assert_array_equal(activity.counts,expected)
            np.testing.assert_allclose(brain.state.v,neurons.v/b.mV,atol=.003,rtol=0)
            np.testing.assert_allclose(brain.state.g,neurons.g/b.mV,atol=.003,rtol=0)


def test_reset_restores_first_step_and_rejects_invalid_indices(mock_graph):
    brain=MaleCNSCore(mock_graph,backend='cpu')
    stimulus=NeuralStimulus(np.array([2]),np.array([30.]))
    first=brain.advance(stimulus,40)
    brain.reset()
    np.testing.assert_array_equal(brain.advance(stimulus,40).counts,first.counts)
    with pytest.raises(ValueError):
        brain.advance(NeuralStimulus(np.array([4]),np.array([30.])),10)
    with pytest.raises(ValueError):
        brain.advance(stimulus,.15)
