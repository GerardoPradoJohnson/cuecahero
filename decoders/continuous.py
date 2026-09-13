"""Continuous neural rate decoder mapping population spike rates to analog actions."""
import math
from typing import Sequence
import numpy as np
from core.contracts import Action, NeuralActivity


class ContinuousRateDecoder:
    """Decodes spike activity into continuous, analog action signals

    using smoothed population firing rates and optional differential pairing.
    """

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        tau_ms: float = 30.0,
        min_rate_hz: float = 0.5,
        max_rate_hz: float = 25.0,
        clip: tuple[float, float] = (0.0, 1.0),
    ):
        if len(groups) == 0 or any(len(g) == 0 for g in groups):
            raise ValueError('Nonempty neuron groups required')
        if not math.isfinite(tau_ms) or tau_ms <= 0.0:
            raise ValueError('tau_ms must be positive and finite')
        self.groups = [np.asarray(g, dtype=np.int32) for g in groups]
        self.tau_ms = tau_ms
        self.min_rate_hz = min_rate_hz
        self.max_rate_hz = max_rate_hz
        self.clip_min, self.clip_max = clip
        self.reset()

    def reset(self) -> None:
        self.smoothed_rates = np.zeros(len(self.groups), dtype=np.float64)
        self.instantaneous_rates = np.zeros(len(self.groups), dtype=np.float64)
        self.values = np.zeros(len(self.groups), dtype=np.float64)

    @property
    def rates(self) -> np.ndarray:
        return self.smoothed_rates

    def decode(self, activity: NeuralActivity) -> Action:
        duration_s = max(1e-6, activity.duration_ms / 1000.0)
        # Compute instantaneous population firing rates (spikes / sec / neuron)
        inst = np.array(
            [activity.counts[g].sum() / (duration_s * len(g)) for g in self.groups],
            dtype=np.float64,
        )
        self.instantaneous_rates = inst

        # Exponential low-pass smoothing across neural step interval
        dt_ms = max(0.1, activity.duration_ms)
        decay = math.exp(-dt_ms / self.tau_ms)
        self.smoothed_rates = self.smoothed_rates * decay + inst * (1.0 - decay)

        # Normalize to continuous [0, 1] range based on dynamic rate window
        rate_span = max(1e-6, self.max_rate_hz - self.min_rate_hz)
        normalized = (self.smoothed_rates - self.min_rate_hz) / rate_span
        clamped = np.clip(normalized, self.clip_min, self.clip_max)
        self.values = clamped
        return Action(tuple(float(x) for x in clamped))
