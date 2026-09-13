from .reward_modulated import RewardModulatedPlasticity

class NoPlasticity:
    """Explicit frozen baseline. Reward is recorded, never injected as dopamine."""
    enabled = False
    def apply(self, neural_state, reward):
        pass

    def status(self, neural_state=None):
        return {'enabled':False, 'model':'none', 'plastic_edges':0, 'changed_edges':0}
