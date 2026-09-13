"""Multimodal sensory fusion encoder combining visual, auditory, and proprioceptive modalities."""
from typing import Sequence
import numpy as np
from core.contracts import Observation, NeuralStimulus, SensorEncoder


class CompositeMultimodalEncoder:
    """Fuses multiple modality-specific encoders into a single unified NeuralStimulus."""

    def __init__(self, encoders: Sequence[SensorEncoder]):
        if not encoders:
            raise ValueError('At least one sensor encoder required')
        self.encoders = list(encoders)

    def reset(self) -> None:
        for enc in self.encoders:
            enc.reset()

    def encode(self, observation: Observation, duration_ms: float) -> NeuralStimulus:
        stimuli = [enc.encode(observation, duration_ms) for enc in self.encoders]
        all_indices = np.concatenate([s.indices for s in stimuli])
        all_currents = np.concatenate([s.currents for s in stimuli])
        return NeuralStimulus(indices=all_indices, currents=all_currents)
