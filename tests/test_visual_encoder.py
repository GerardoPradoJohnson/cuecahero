import numpy as np
import pytest
from core.contracts import Observation
from sensors import VisualEncoder

def test_encoder_samples_pixels_and_reset_discards_filter_history():
    encoder=VisualEncoder(np.array([2,3]),np.array([[0,0],[1,1]]),np.array([0]))
    pixels=np.zeros((2,2,3),dtype=np.uint8)
    pixels[1,1]=255
    stimulus=encoder.encode(Observation(pixels,0),10)
    assert stimulus.indices.tolist()==[0,2,3]
    assert stimulus.currents[0]==12 and stimulus.currents[1]==0 and stimulus.currents[2]>0
    encoder.reset()
    np.testing.assert_array_equal(encoder.encode(Observation(pixels,0),10).currents,stimulus.currents)
    with pytest.raises(ValueError):
        encoder.encode(Observation(pixels,0),float('nan'))
