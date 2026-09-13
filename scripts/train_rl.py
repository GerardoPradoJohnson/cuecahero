"""Closed-loop Reinforcement Learning training for the Cueca Hero neural readout.

Trains the external reader using policy gradient on closed-loop RewardSignal.
Operates on causal neural population features from lamina interneurons.
Supports atomic checkpoints, deterministic resuming, and independent evaluation seeds.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.contracts import Action
from environments import CuecaHeroEnvironment
from training.rl import ReinforcementReadoutTrainer


def temporal_features_batch(raw, lags):
    """Construct lagged feature matrix from full raw feature sequence."""
    raw = np.asarray(raw, dtype=np.float64)
    parts = []
    for lag in lags:
        shifted = np.zeros_like(raw)
        if lag < len(raw):
            shifted[lag:] = raw[:len(raw) - lag]
        parts.append(shifted)
    return np.concatenate(parts, axis=1)


def run_episode_cached(trainer, seed, raw_features, rng, explore=True):
    """Run one closed-loop game episode using pre-extracted neural features."""
    env = CuecaHeroEnvironment()
    env.reset(seed)
    trainer.reset_runtime()
    
    X = temporal_features_batch(raw_features, trainer.lags)
    Z = (X - trainer.mean) / trainer.scale
    
    trajectory = []
    rewards_by_lane = []
    
    for t in range(len(Z)):
        norm_x = Z[t]
        current_time_ms = env.time * 1000.0
        action, rec = trainer.step(norm_x, current_time_ms, explore=explore, rng=rng)
        trajectory.append(rec)
        
        prev_hits = [l['hits'] for l in env.lanes]
        prev_wrong = [l['wrong'] for l in env.lanes]
        prev_misses = [l['misses'] for l in env.lanes]
        
        env.step(action, 1.0 / 30.0)
        
        step_r = np.zeros(4, dtype=np.float64)
        for lane in range(4):
            dh = env.lanes[lane]['hits'] - prev_hits[lane]
            dw = env.lanes[lane]['wrong'] - prev_wrong[lane]
            dm = env.lanes[lane]['misses'] - prev_misses[lane]
            if dh > 0:
                ev = env.lanes[lane].get('last_event')
                step_r[lane] += (1.0 if ev and ev.get('kind') == 'perfect' else 0.6) * dh
            if dw > 0:
                step_r[lane] -= 0.1 * dw
            if dm > 0:
                step_r[lane] -= 1.0 * dm
        rewards_by_lane.append(step_r)
        
    grad_norm = 0.0
    if explore:
        grad_norm = trainer.update_policy(trajectory, np.asarray(rewards_by_lane))
        
    state = env.get_state()
    total_reward = float(np.sum(rewards_by_lane))
    return state, total_reward, grad_norm


def run_episode_live(trainer, seed, brain, encoder, rng, explore=True):
    """Run one closed-loop game episode live with MaleCNS and visual encoder."""
    env = CuecaHeroEnvironment()
    env.reset(seed)
    brain.reset()
    encoder.reset()
    trainer.reset_runtime()
    
    trajectory = []
    rewards_by_lane = []
    
    while not env.is_done():
        stimulus = encoder.encode(env.get_observation(), 10.0)
        activity = brain.advance(stimulus, 10.0)
        norm_x = trainer.extract_features(activity)
        current_time_ms = env.time * 1000.0
        
        action, rec = trainer.step(norm_x, current_time_ms, explore=explore, rng=rng)
        trajectory.append(rec)
        
        prev_hits = [l['hits'] for l in env.lanes]
        prev_wrong = [l['wrong'] for l in env.lanes]
        prev_misses = [l['misses'] for l in env.lanes]
        
        env.step(action, 1.0 / 30.0)
        
        step_r = np.zeros(4, dtype=np.float64)
        for lane in range(4):
            dh = env.lanes[lane]['hits'] - prev_hits[lane]
            dw = env.lanes[lane]['wrong'] - prev_wrong[lane]
            dm = env.lanes[lane]['misses'] - prev_misses[lane]
            if dh > 0:
                ev = env.lanes[lane].get('last_event')
                step_r[lane] += (1.0 if ev and ev.get('kind') == 'perfect' else 0.6) * dh
            if dw > 0:
                step_r[lane] -= 0.1 * dw
            if dm > 0:
                step_r[lane] -= 1.0 * dm
        rewards_by_lane.append(step_r)
        
    grad_norm = 0.0
    if explore:
        grad_norm = trainer.update_policy(trajectory, np.asarray(rewards_by_lane))
        
    state = env.get_state()
    total_reward = float(np.sum(rewards_by_lane))
    return state, total_reward, grad_norm


def main():
    parser = argparse.ArgumentParser(description='Train Cueca Hero readout with closed-loop RL.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--base-model', type=Path, help='Initial model template JSON')
    source.add_argument('--resume', type=Path, help='Checkpoint NPZ to resume from')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for checkpoints and models')
    parser.add_argument('--episodes', type=int, default=10, help='Total target episodes to train')
    parser.add_argument('--seeds', type=int, nargs='+', default=[211, 307, 401, 503, 601, 701],
                        help='Training seeds to cycle through')
    parser.add_argument('--episodes-dir', type=Path, default=None,
                        help='Optional path to directory with pre-extracted features (e.g. outputs/calibration/perception-v4)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--explore-temp', type=float, default=0.25, help='Exploration temperature')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--rng-seed', type=int, default=42, help='RNG seed for reproducible exploration')
    parser.add_argument('--threshold', type=float, default=None, help='Readout firing threshold (defaults to model template)')
    args = parser.parse_args()

    if args.resume is None and args.output.exists():
        parser.error(f'Output directory {args.output} already exists. Choose a new directory.')
    args.output.mkdir(parents=True, exist_ok=True)

    if args.resume:
        trainer = ReinforcementReadoutTrainer.load(args.resume)
        if args.threshold is not None:
            trainer.threshold = float(args.threshold)
        print(f'Resumed trainer from {args.resume} (consumed {len(trainer.episodes)} episodes)', flush=True)
    else:
        model_data = json.loads(args.base_model.read_text())
        trainer = ReinforcementReadoutTrainer(
            model=model_data,
            lr=args.lr,
            gamma=args.gamma,
            explore_temp=args.explore_temp
        )
        if args.threshold is not None:
            trainer.threshold = float(args.threshold)

    consumed_count = len(trainer.episodes)
    if consumed_count >= args.episodes:
        print(f'Target {args.episodes} episodes already reached ({consumed_count} consumed). Nothing to do.')
        return

    # Check for cached neural features
    cached_features = {}
    if args.episodes_dir and args.episodes_dir.exists():
        for p in args.episodes_dir.glob('training-*.npz'):
            try:
                seed_str = p.stem.split('-')[-1]
                s_val = int(seed_str)
                with np.load(p, allow_pickle=False) as d:
                    cached_features[s_val] = d['features']
            except Exception:
                pass

    brain = None
    encoder = None
    need_live = any(s not in cached_features for s in args.seeds)
    if need_live:
        from core.malecns import MaleCNSCore
        from sensors.contrast import ContrastVisualEncoder
        print('Initializing MaleCNSCore for live training...', flush=True)
        brain = MaleCNSCore()
        encoder = ContrastVisualEncoder(
            brain.state.retina, brain.state.uv, brain.state.lamina,
            **trainer.template['sensor']
        )

    rng = np.random.default_rng(args.rng_seed + consumed_count)
    started = time.perf_counter()

    for idx in range(consumed_count, args.episodes):
        ep_seed = args.seeds[idx % len(args.seeds)]
        t0 = time.perf_counter()
        
        if ep_seed in cached_features:
            state, total_reward, grad_norm = run_episode_cached(
                trainer, ep_seed, cached_features[ep_seed], rng, explore=True
            )
        else:
            state, total_reward, grad_norm = run_episode_live(
                trainer, ep_seed, brain, encoder, rng, explore=True
            )
            
        elapsed = time.perf_counter() - t0
        hits_by_lane = [l['hits'] for l in state['lanes']]
        wrong_by_lane = [l['wrong'] for l in state['lanes']]
        misses_by_lane = [l['misses'] for l in state['lanes']]
        
        ep_record = {
            'index': idx + 1,
            'name': f'episode-{idx + 1:04d}',
            'seed': ep_seed,
            'score': state['score'],
            'hits': state['hits'],
            'misses': state['misses'],
            'wrong': state['wrong'],
            'accuracy': state['accuracy'],
            'reward': round(total_reward, 2),
            'grad_norm': round(grad_norm, 4),
            'hits_by_lane': hits_by_lane,
            'wrong_by_lane': wrong_by_lane,
            'misses_by_lane': misses_by_lane,
            'wall_seconds': round(elapsed, 2)
        }
        
        ckpt_path = args.output / f"episode-{idx + 1:04d}.npz"
        ep_record['sha256'] = ckpt_path.name
        trainer.episodes.append(ep_record)
        digest = trainer.save(ckpt_path)
        ep_record['sha256'] = digest
        
        print(f"[Episode {idx + 1:04d}/{args.episodes}] Seed={ep_seed} Score={state['score']:4d} "
              f"Hits={state['hits']:2d}/48 Misses={state['misses']:2d} Wrong={state['wrong']:2d} "
              f"Reward={total_reward:6.1f} GradNorm={grad_norm:6.4f} ({elapsed:.1f}s)", flush=True)

    # Save final model and training summary progress
    final_model = trainer.model()
    (args.output / 'model.json').write_text(json.dumps(final_model, indent=2) + '\n')
    
    progress = {
        'kind': 'reinforcement_external_readout',
        'algorithm': 'closed_loop_policy_gradient_with_advantage',
        'episodes_completed': len(trainer.episodes),
        'episodes': trainer.episodes,
        'wall_seconds': round(time.perf_counter() - started, 2),
        'provenance': {
            'base_model_graph_sha256': trainer.template.get('graph_sha256'),
            'training_seeds': args.seeds
        },
        'limits': 'Closed-loop RL on external readout policy. Does not alter MaleCNS synaptic weights or represent biological plasticity.'
    }
    (args.output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    print(f"Training completed. Exported model to {args.output / 'model.json'}", flush=True)


if __name__ == '__main__':
    main()
