"""Four-leg cueca stepping embodiment translating descending motor activity into physical leg strikes."""
import math
from typing import Sequence
import numpy as np
from core.contracts import Action


class LegSteppingEmbodiment:
    """Models the 4-leg cueca zapateo motor system.
    
    Maps descending motor commands for the 4 legs (T1-L, T2-L, T2-R, T1-R)
    to physical foot strikes against floor tap pads.
    """
    LEG_NAMES = ('T1-L', 'T2-L', 'T2-R', 'T1-R')
    LANES = ('D', 'F', 'J', 'K')

    def __init__(self, strike_threshold: float = 0.5, impact_stiffness: float = 120.0):
        self.strike_threshold = float(strike_threshold)
        self.impact_stiffness = float(impact_stiffness)
        self.leg_positions = np.zeros(4, dtype=np.float64)  # 0.0 = contact, >0 = lifted
        self.leg_forces = np.zeros(4, dtype=np.float64)
        self.last_actions = np.zeros(4, dtype=np.float64)

    def apply(self, action: Action) -> Action:
        """Processes descending commands into physical leg impacts on pads."""
        if len(action.values) != 4 or not all(math.isfinite(x) for x in action.values):
            raise ValueError('LegSteppingEmbodiment requires 4 finite action commands')

        raw_values = np.asarray(action.values, dtype=np.float64)
        # Leg strike happens when motor drive crosses threshold
        strikes = raw_values >= self.strike_threshold

        # Update simulated contact forces on pads
        self.leg_forces = np.where(strikes, raw_values * self.impact_stiffness, 0.0)
        self.last_actions = strikes.astype(np.float64)

        return Action(tuple(float(x) for x in strikes))

    def get_contact_telemetry(self) -> dict:
        """Returns leg impact forces and state for telemetry/proprioception."""
        return {
            'legs': {
                name: {
                    'lane': lane,
                    'force_mN': float(force),
                    'striking': bool(strike),
                }
                for name, lane, force, strike in zip(
                    self.LEG_NAMES, self.LANES, self.leg_forces, self.last_actions
                )
            },
            'total_impact_mN': float(np.sum(self.leg_forces)),
        }
