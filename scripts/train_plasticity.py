"""Closed-loop synaptic plasticity training for MaleCNS on Cueca Hero.

Applies bounded reward-modulated Hebbian plasticity to the 4,276 identified
synaptic connections between visual retina receptors and lamina interneurons.
Synaptic weights persist and evolve across episodes; fast state resets per episode.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.malecns import MaleCNSCore
from core.paths import DATA, GRAPH, ROOT
from decoders.calibrated import CalibratedDecoder
from environments import CuecaHeroEnvironment
from plasticity.reward_modulated import RewardModulatedPlasticity
from sensors.contrast import ContrastVisualEncoder


def verify_graph():
    """Verify runtime graph matches completed data audit."""
    expected = json.loads((DATA / 'outputs/doom/audit/data-integrity.json').read_text())
    digest = hashlib.sha256()
    with GRAPH.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    graph_hash = digest.hexdigest()
    if not expected['passed'] or graph_hash != expected['graph_sha256']:
        raise ValueError('Graph does not match verified data audit')
    return graph_hash


def main():
    parser = argparse.ArgumentParser(description='Train MaleCNS synaptic plasticity on Cueca Hero.')
    parser.add_argument('--config', type=Path, default=Path('experiments/cueca_hero_plastic.json'),
                        help='Experiment configuration JSON')
    parser.add_argument('--episodes', type=int, default=10, help='Total episodes to train')
    parser.add_argument('--seeds', type=int, nargs='+', default=[211, 307, 401, 503, 601, 701],
                        help='Training seeds to cycle through')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for checkpoints')
    parser.add_argument('--resume', type=Path, default=None, help='Checkpoint NPZ to resume from')
    args = parser.parse_args()

    if args.resume is None and args.output.exists():
        parser.error(f'Output directory {args.output} already exists. Choose a new directory.')
    args.output.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text())
    p_cfg = config.get('plasticity', {})
    if not p_cfg.get('enabled') or p_cfg.get('type') != 'reward_modulated':
        parser.error('Configuration must enable reward_modulated plasticity')

    print('Verifying MaleCNS connectome integrity...', flush=True)
    graph_sha256 = verify_graph()

    print('Initializing MaleCNSCore, ContrastVisualEncoder and CalibratedDecoder...', flush=True)
    brain = MaleCNSCore(backend=config['brain'].get('backend', 'auto'))
    
    sensor_cfg = config['sensors'][0]
    encoder = ContrastVisualEncoder(
        retina=brain.state.retina,
        uv=brain.state.uv,
        lamina=brain.state.lamina,
        **{k: v for k, v in sensor_cfg.items() if k != 'type'}
    )
    
    decoder_cfg = config['decoder']
    model_path = ROOT / decoder_cfg['model'] if not Path(decoder_cfg['model']).is_absolute() else Path(decoder_cfg['model'])
    decoder_model = json.loads(model_path.read_text())
    decoder = CalibratedDecoder(decoder_model)

    plasticity = RewardModulatedPlasticity(
        eta=p_cfg.get('eta', 0.002),
        eligibility_tau_ms=p_cfg.get('eligibility_tau_ms', 750.0),
        minimum_fraction=p_cfg.get('minimum_fraction', 0.95),
        maximum_fraction=p_cfg.get('maximum_fraction', 1.05)
    )
    
    # Bind directly to the 4,276 synapses connecting sensory retina to lamina interneurons
    plasticity.bind(brain, encoder.retina, decoder.indices, graph_sha256)
    print(f"Bound RewardModulatedPlasticity to {len(plasticity.edges)} real synaptic connections.", flush=True)

    consumed = 0
    episodes_history = []
    
    if args.resume:
        print(f"Restoring synaptic weights and eligibility from {args.resume}...", flush=True)
        plasticity.restore(brain, args.resume)
        consumed_str = args.resume.stem.split('-')[-1]
        try:
            consumed = int(consumed_str)
        except ValueError:
            consumed = 1
        print(f"Restored checkpoint: {plasticity.status(brain)['changed_edges']} modified edges.", flush=True)

    started = time.perf_counter()

    for idx in range(consumed, args.episodes):
        ep_seed = args.seeds[idx % len(args.seeds)]
        t0 = time.perf_counter()
        
        env = CuecaHeroEnvironment(
            bpm=config['environment'].get('bpm', 108),
            bars=config['environment'].get('bars', 12)
        )
        env.reset(ep_seed)
        brain.reset()  # Resets fast state (voltages, refractory counts) while preserving synaptic weights
        encoder.reset()
        decoder.reset()
        
        total_reward = 0.0
        
        while not env.is_done():
            stimulus = encoder.encode(env.get_observation(), 10.0)
            activity = brain.advance(stimulus, 10.0)
            action = decoder.decode(activity)
            env.step(action, 1.0 / 30.0)
            
            r_sig = env.get_reward()
            total_reward += r_sig.magnitude
            plasticity.apply(brain, r_sig)
            
        elapsed = time.perf_counter() - t0
        state = env.get_state()
        p_status = plasticity.status(brain)
        
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
            'changed_edges': p_status['changed_edges'],
            'mean_fraction': round(p_status['mean_fraction'], 6),
            'min_fraction': round(p_status['minimum_fraction'], 6),
            'max_fraction': round(p_status['maximum_fraction'], 6),
            'updates': p_status['updates'],
            'wall_seconds': round(elapsed, 2)
        }
        
        ckpt_path = args.output / f"episode-{idx + 1:04d}.npz"
        digest = plasticity.checkpoint(brain, ckpt_path)
        ep_record['sha256'] = digest
        episodes_history.append(ep_record)
        
        print(f"[Episode {idx + 1:04d}/{args.episodes}] Seed={ep_seed} Score={state['score']:4d} "
              f"Hits={state['hits']:2d}/48 Misses={state['misses']:2d} Wrong={state['wrong']:2d} "
              f"Reward={total_reward:6.1f} ChangedEdges={p_status['changed_edges']}/{len(plasticity.edges)} "
              f"MeanFraction={p_status['mean_fraction']:.4f} ({elapsed:.1f}s)", flush=True)

    summary = {
        'kind': 'synaptic_plasticity_training',
        'graph_sha256': graph_sha256,
        'plastic_edges_total': len(plasticity.edges),
        'final_status': plasticity.status(brain),
        'episodes_completed': len(episodes_history),
        'episodes': episodes_history,
        'wall_seconds': round(time.perf_counter() - started, 2),
        'limits': 'Bounded reward-modulated Hebbian plasticity on existing MaleCNS connections. Engineering hypothesis, not complete biological dopamine.'
    }
    (args.output / 'progress.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"\nPlasticity training complete. Progress saved to {args.output / 'progress.json'}", flush=True)


if __name__ == '__main__':
    main()
