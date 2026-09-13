"""Unit tests for the four-leg cueca zapateo embodiment."""
import pytest
from core.contracts import Action
from embodiments import LegSteppingEmbodiment


def test_leg_stepping_strike_threshold():
    emb = LegSteppingEmbodiment(strike_threshold=0.5, impact_stiffness=100.0)
    
    # Below threshold (no strike)
    action = Action((0.2, 0.49, 0.1, 0.0))
    result = emb.apply(action)
    assert result.values == (0.0, 0.0, 0.0, 0.0)
    
    # Above threshold (strikes on D, F, K)
    action = Action((0.8, 0.5, 0.3, 0.95))
    result = emb.apply(action)
    assert result.values == (1.0, 1.0, 0.0, 1.0)
    
    # Check telemetry
    telem = emb.get_contact_telemetry()
    assert telem['legs']['T1-L']['striking'] is True
    assert telem['legs']['T1-L']['force_mN'] == pytest.approx(80.0)
    assert telem['legs']['T2-L']['striking'] is True
    assert telem['legs']['T2-L']['force_mN'] == pytest.approx(50.0)
    assert telem['legs']['T2-R']['striking'] is False
    assert telem['legs']['T2-R']['force_mN'] == pytest.approx(0.0)
    assert telem['legs']['T1-R']['striking'] is True
    assert telem['legs']['T1-R']['force_mN'] == pytest.approx(95.0)
    assert telem['total_impact_mN'] == pytest.approx(225.0)


def test_leg_stepping_invalid_dimension():
    emb = LegSteppingEmbodiment()
    with pytest.raises(ValueError, match="requires 4 finite action commands"):
        emb.apply(Action((0.5, 0.5)))
