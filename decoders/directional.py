"""Fixed engineering readouts; no access to observations or the note chart."""
import numpy as np
from core.contracts import Action

class DirectionalDecoder:
    def __init__(self, groups, threshold_hz=1., cooldown_ms=120.):
        if len(groups) != 4 or any(len(group) == 0 for group in groups):
            raise ValueError('Four nonempty readout groups required')
        self.groups = [np.asarray(g, dtype=np.int32) for g in groups]
        self.threshold_hz, self.cooldown_ms = threshold_hz, cooldown_ms
        self.reset()

    def reset(self):
        self.last = np.full(4, -1e12)
        self.rates = np.zeros(4)

    def decode(self, activity):
        self.rates = np.array([activity.counts[g].sum() * 1000 / activity.duration_ms / len(g) for g in self.groups])
        fire = (self.rates >= self.threshold_hz) & (activity.timestamp_ms - self.last >= self.cooldown_ms)
        self.last[fire] = activity.timestamp_ms
        return Action(tuple(float(x) for x in fire))
