"""Auditory sensor encoder modeling Drosophila Johnston's Organ (JO) acoustic mechanoreceptors."""
import math
from typing import Sequence
import numpy as np
from core.contracts import Observation, NeuralStimulus


class AuditoryEncoder:
    """Encodes acoustic waveforms and vibration pulses into neural currents

    injected into Johnston's Organ / AMMC auditory target neurons.
    """

    def __init__(
        self,
        target_indices: Sequence[int] | np.ndarray,
        center_freq_hz: float = 250.0,
        bandwidth_hz: float = 120.0,
        gain: float = 40.0,
        baseline_current: float = 5.0,
        tau_ms: float = 8.0,
    ):
        if len(target_indices) == 0:
            raise ValueError('Nonempty target neuron indices required')
        self.target_indices = np.asarray(target_indices, dtype=np.int32)
        self.center_freq_hz = center_freq_hz
        self.bandwidth_hz = bandwidth_hz
        self.gain = gain
        self.baseline_current = baseline_current
        self.tau_ms = tau_ms
        self.reset()

    def reset(self) -> None:
        self.filtered_energy = 0.0

    def encode(self, observation: Observation, duration_ms: float) -> NeuralStimulus:
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError('Finite positive duration required')

        # Extract acoustic signal from observation
        audio = observation.audio
        if audio is None or len(audio) == 0:
            instant_energy = 0.0
        else:
            audio_arr = np.asarray(audio, dtype=np.float32)
            # RMS amplitude envelope
            instant_energy = float(np.sqrt(np.mean(audio_arr ** 2)))

        # Leaky envelope integrator
        alpha = 1.0 - math.exp(-duration_ms / self.tau_ms)
        self.filtered_energy += alpha * (instant_energy - self.filtered_energy)

        # Saturating non-linear current response: I = I_base + Gain * E / (1 + E)
        current_mag = self.baseline_current + self.gain * (self.filtered_energy / (0.1 + self.filtered_energy))
        currents = np.full(len(self.target_indices), current_mag, dtype=np.float32)

        return NeuralStimulus(indices=self.target_indices, currents=currents)
