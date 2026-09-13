"""Proprioceptive and tactile sensor encoder modeling fly mechanosensory chordotonal organs."""
import math
from typing import Sequence
import numpy as np
from core.contracts import Observation, NeuralStimulus


class ProprioceptiveEncoder:
    """Encodes velocity, angular acceleration, and collision impacts into

    phasic-tonic neural currents for mechanosensory target neurons.
    """

    def __init__(
        self,
        target_indices: Sequence[int] | np.ndarray,
        velocity_gain: float = 0.5,
        angular_gain: float = 4.0,
        impact_gain: float = 35.0,
        tonic_bias: float = 6.0,
        phasic_tau_ms: float = 15.0,
    ):
        if len(target_indices) == 0:
            raise ValueError('Nonempty target neuron indices required')
        self.target_indices = np.asarray(target_indices, dtype=np.int32)
        self.velocity_gain = velocity_gain
        self.angular_gain = angular_gain
        self.impact_gain = impact_gain
        self.tonic_bias = tonic_bias
        self.phasic_tau_ms = phasic_tau_ms
        self.reset()

    def reset(self) -> None:
        self.last_velocity = 0.0
        self.last_angle = 0.0
        self.phasic_current = 0.0

    def encode(self, observation: Observation, duration_ms: float) -> NeuralStimulus:
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError('Finite positive duration required')

        prop = observation.proprioception or {}
        speed = float(prop.get('speed', 0.0))
        angular_vel = float(prop.get('angular_velocity', 0.0))
        collision = float(prop.get('collision', 0.0))

        # Phasic response: sudden changes in speed, sharp turns, or direct wall/obstacle impact
        accel = abs(speed - self.last_velocity) / max(1e-3, duration_ms / 1000.0)
        self.last_velocity = speed

        instant_phasic = (accel * 0.05 + abs(angular_vel) * self.angular_gain + collision * self.impact_gain)
        alpha = 1.0 - math.exp(-duration_ms / self.phasic_tau_ms)
        self.phasic_current = self.phasic_current * (1.0 - alpha) + instant_phasic * alpha

        # Tonic response: steady speed sensation
        tonic_current = self.tonic_bias + speed * self.velocity_gain

        total_current = float(tonic_current + self.phasic_current)
        currents = np.full(len(self.target_indices), total_current, dtype=np.float32)

        return NeuralStimulus(indices=self.target_indices, currents=currents)
