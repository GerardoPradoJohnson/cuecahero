"""Deterministic episodic reinforcement learning trainer with verifiable checkpoints.

Learns directly from closed-loop Cueca Hero RewardSignal.
Operates on causal neural population features from lamina interneurons.
Maintains full provenance, moving advantage baselines, Adam optimizer state,
and verifiable SHA-256 checkpoints.
"""
import copy
import hashlib
import io
import json
import math
import os
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

from core.contracts import Action


def temporal_features(history, lags):
    """Concatenate lagged feature vectors from a history deque."""
    parts = []
    for lag in lags:
        if lag < len(history):
            parts.append(history[lag])
        else:
            parts.append(history[-1])
    return np.concatenate(parts)


def state_digest(metadata, weights, bias, m, v):
    """Compute deterministic SHA-256 digest of trainer state."""
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for array in (weights, bias, m, v):
        digest.update(np.ascontiguousarray(array, dtype='<f8').tobytes())
    return digest.hexdigest()


class ReinforcementReadoutTrainer:
    """Episodic Reinforcement Learning policy trainer for Cueca Hero.
    
    Optimizes weights and biases of the readout policy using policy gradient
    with per-lane credit assignment and moving baseline.
    """
    def __init__(self, model, lr=1e-3, gamma=0.95, explore_temp=0.25, l2_reg=1e-4):
        self.template = copy.deepcopy(model)
        self.groups = [np.asarray(g, dtype=np.int32) for g in self.template.get('groups', [[i] for i in self.template['indices']])]
        self.indices = np.unique(np.concatenate(self.groups))
        self.feature_count = len(self.groups)
        self.lags = self.template.get('lags', [0])
        self.dim = self.feature_count * len(self.lags)
        
        self.mean = np.asarray(self.template['mean'], dtype=np.float64)
        self.scale = np.asarray(self.template['scale'], dtype=np.float64)
        self.weights = np.asarray(self.template['weights'], dtype=np.float64).copy()
        self.bias = np.asarray(self.template['bias'], dtype=np.float64).copy()
        self.base_weights = self.weights.copy()
        self.base_bias = self.bias.copy()
        
        self.threshold = float(self.template.get('threshold', 0.45))
        self.release = float(self.template.get('release', 0.20))
        self.tau_ms = float(self.template['tau_ms'])
        self.cooldown_ms = float(self.template.get('cooldown_ms', 30.0))
        
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.explore_temp = float(explore_temp)
        self.l2_reg = float(l2_reg)
        
        # Adam optimizer parameters & moments
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.m = np.zeros((self.dim + 1, 4), dtype=np.float64)
        self.v = np.zeros((self.dim + 1, 4), dtype=np.float64)
        self.opt_step = 0
        
        # Moving baseline for advantage estimation per lane
        self.baseline_return = np.zeros(4, dtype=np.float64)
        self.baseline_count = 0
        
        self.episodes = []
        self.reset_runtime()

    def reset_runtime(self):
        """Reset online filtering and refractory state for a new episode."""
        self.filtered = np.zeros(self.feature_count, dtype=np.float64)
        self.history = deque([self.filtered.copy() for _ in range(max(self.lags) + 1)], maxlen=max(self.lags) + 1)
        self.armed = np.ones(4, dtype=bool)
        self.last_press_time_ms = np.full(4, -1e12, dtype=np.float64)
        self.last_rates = np.zeros(4, dtype=np.float64)
        self.held_actions = np.zeros(4, dtype=bool)

    def extract_features(self, activity):
        """Update causal exponential filter and return normalized lagged features."""
        alpha = 1.0 - math.exp(-activity.duration_ms / self.tau_ms)
        rates = np.array([activity.counts[g].mean() for g in self.groups]) * 1000.0 / activity.duration_ms
        self.filtered += alpha * (rates - self.filtered)
        self.history.appendleft(self.filtered.copy())
        raw_x = temporal_features(self.history, self.lags)
        norm_x = (raw_x - self.mean) / self.scale
        return norm_x

    def step(self, norm_x, current_time_ms, explore=True, rng=None):
        """Compute action logits, sample or evaluate actions, and record step data.
        
        Returns:
            action: Action instance with 4 binary values (0.0 or 1.0)
            step_record: dict containing state needed for policy gradient update
        """
        if rng is None:
            rng = np.random.default_rng()
            
        logits = norm_x @ self.weights + self.bias  # shape (4,)
        if self.template.get('action_mode') == 'held_sigmoid':
            if explore:
                raise ValueError('This supervised checkpoint is for frozen evaluation; retrain offline.')
            rates = 1. / (1. + np.exp(-np.clip(logits, -30, 30)))
            self.held_actions = (rates >= self.threshold) | (self.held_actions & (rates >= self.release))
            self.last_rates = rates.copy()
            return Action(tuple(float(v) for v in self.held_actions)), {}
        rates = np.clip(logits, 0.0, 1.0)
        self.last_rates = rates.copy()
        
        # Armed updates when rate drops below release threshold
        self.armed |= (rates < self.release)
        
        cooldown_ok = (current_time_ms - self.last_press_time_ms >= self.cooldown_ms)
        eligible = self.armed & cooldown_ok
        
        actions = np.zeros(4, dtype=np.float64)
        action_probs = np.zeros(4, dtype=np.float64)
        
        for lane in range(4):
            if not eligible[lane]:
                actions[lane] = 0.0
                action_probs[lane] = 0.0
                continue
                
            if explore:
                # Stochastic sigmoid policy over logit excess above threshold
                u = (logits[lane] - self.threshold) / self.explore_temp
                u = np.clip(u, -20.0, 20.0)
                prob = 1.0 / (1.0 + math.exp(-u))
                action_probs[lane] = prob
                actions[lane] = 1.0 if rng.random() < prob else 0.0
            else:
                # Deterministic inference: fire if rate >= threshold
                fire = rates[lane] >= self.threshold
                actions[lane] = 1.0 if fire else 0.0
                action_probs[lane] = 1.0 if fire else 0.0

        for lane in range(4):
            if actions[lane] > 0.5:
                self.last_press_time_ms[lane] = current_time_ms
                self.armed[lane] = False
                
        step_record = {
            'x': norm_x,
            'eligible': eligible,
            'logits': logits,
            'probs': action_probs,
            'actions': actions
        }
        return Action(tuple(actions)), step_record

    def update_policy(self, trajectory, rewards_by_lane):
        """Apply policy gradient update with per-lane advantage credit assignment.
        
        Args:
            trajectory: list of step_record dicts
            rewards_by_lane: (T, 4) ndarray of rewards received at each timestep
        """
        T = len(trajectory)
        if T == 0:
            return 0.0
            
        # Compute discounted returns per lane
        returns = np.zeros((T, 4), dtype=np.float64)
        running = np.zeros(4, dtype=np.float64)
        for t in reversed(range(T)):
            running = rewards_by_lane[t] + self.gamma * running
            returns[t] = running

        # Update moving baseline per lane
        episode_mean_return = returns[0]
        if self.baseline_count == 0:
            self.baseline_return = episode_mean_return.copy()
        else:
            self.baseline_return = 0.9 * self.baseline_return + 0.1 * episode_mean_return
        self.baseline_count += 1
        
        # Policy gradient accumulation
        grad = np.zeros((self.dim + 1, 4), dtype=np.float64)
        
        for t in range(T):
            rec = trajectory[t]
            x_aug = np.append(rec['x'], 1.0)  # shape (dim + 1,)
            eligible = rec['eligible']
            probs = rec['probs']
            actions = rec['actions']
            advantage = returns[t] - self.baseline_return
            
            for lane in range(4):
                if eligible[lane]:
                    # d log pi / d theta = (action - prob) / explore_temp * x_aug
                    score_grad = (actions[lane] - probs[lane]) / self.explore_temp * x_aug
                    grad[:, lane] += advantage[lane] * score_grad
                    
        # Average gradient over episode length
        grad /= max(1, T)
        
        # Add L2 weight decay toward initial baseline weights
        curr_theta = np.vstack([self.weights, self.bias])
        base_theta = np.vstack([self.base_weights, self.base_bias])
        grad -= self.l2_reg * (curr_theta - base_theta)
        
        # Adam optimizer step (maximize reward -> gradient ascent)
        self.opt_step += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        
        m_hat = self.m / (1.0 - self.beta1 ** self.opt_step)
        v_hat = self.v / (1.0 - self.beta2 ** self.opt_step)
        
        step_theta = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        new_theta = curr_theta + step_theta
        
        self.weights = new_theta[:-1]
        self.bias = new_theta[-1]
        
        grad_norm = float(np.linalg.norm(grad))
        return grad_norm

    def model(self):
        """Export trained policy parameters formatted as a verified CalibratedDecoder model."""
        result = copy.deepcopy(self.template)
        if result.get('action_mode') == 'held_sigmoid':
            training = result.get(
                'training_method',
                f'Supervised external neural readout; {len(self.episodes)} offline epochs. '
                'No MaleCNS synaptic plasticity.'
            )
        else:
            training = f'Closed-loop RL trained readout; {len(self.episodes)} episodes. No MaleCNS plasticity.'
        result.update(
            weights=self.weights.tolist(),
            bias=self.bias.tolist(),
            threshold=self.threshold,
            release=self.release,
            cooldown_ms=self.cooldown_ms,
            training=training
        )
        result['rl_episodes_sha256'] = {e['name']: e.get('sha256', '') for e in self.episodes}
        result.pop('threshold_selection', None)
        result.pop('training_wall_seconds', None)
        return result

    def save(self, path):
        """Atomically save checkpoint with verifiable SHA-256 digest."""
        path = Path(path)
        metadata = {
            'schema': 1,
            'kind': 'reinforcement_external_readout',
            'template': self.template,
            'lr': self.lr,
            'gamma': self.gamma,
            'explore_temp': self.explore_temp,
            'l2_reg': self.l2_reg,
            'threshold': self.threshold,
            'release': self.release,
            'cooldown_ms': self.cooldown_ms,
            'opt_step': self.opt_step,
            'baseline_return': self.baseline_return.tolist(),
            'baseline_count': self.baseline_count,
            'episodes': self.episodes
        }
        checksum = state_digest(metadata, self.weights, self.bias, self.m, self.v)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            metadata=json.dumps(metadata, allow_nan=False),
            weights=self.weights,
            bias=self.bias,
            m=self.m,
            v=self.v,
            sha256=checksum
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.rl-ckpt-', delete=False) as stream:
                temp_file = Path(stream.name)
                stream.write(buffer.getvalue())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_file, path)
        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)
        return checksum

    @classmethod
    def load(cls, path):
        """Restore exact trainer state from verified checkpoint."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            weights = np.asarray(data['weights'], dtype=np.float64)
            bias = np.asarray(data['bias'], dtype=np.float64)
            m = np.asarray(data['m'], dtype=np.float64)
            v = np.asarray(data['v'], dtype=np.float64)
            checksum = str(data['sha256'])
            
        expected = state_digest(metadata, weights, bias, m, v)
        if checksum != expected:
            raise ValueError('RL checkpoint checksum mismatch or corrupted file')
            
        if metadata.get('schema') != 1 or metadata.get('kind') != 'reinforcement_external_readout':
            raise ValueError('Unknown or incompatible RL checkpoint format')
            
        trainer = cls(
            model=metadata['template'],
            lr=metadata['lr'],
            gamma=metadata['gamma'],
            explore_temp=metadata['explore_temp'],
            l2_reg=metadata['l2_reg']
        )
        trainer.weights = weights.copy()
        trainer.bias = bias.copy()
        trainer.m = m.copy()
        trainer.v = v.copy()
        trainer.opt_step = int(metadata['opt_step'])
        trainer.threshold = float(metadata['threshold'])
        trainer.release = float(metadata['release'])
        trainer.cooldown_ms = float(metadata['cooldown_ms'])
        trainer.baseline_return = np.asarray(metadata['baseline_return'], dtype=np.float64)
        trainer.baseline_count = int(metadata['baseline_count'])
        trainer.episodes = copy.deepcopy(metadata['episodes'])
        return trainer
