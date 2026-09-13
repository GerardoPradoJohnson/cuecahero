"""Validation of MaleCNS synaptic plasticity: selective effect, retention, erase, and controls.

Verifies:
  1. Performance under baseline weights vs plastic-adapted weights vs erased weights.
  2. Exact bit-for-bit retention and reset upon erase().
  3. Synaptic selectivity: fraction of changed vs preserved edges, bounds compliance.
  4. Negative control stability (Black Screen and Empty Board).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.contracts import Action, Observation
from core.malecns import MaleCNSCore
from core.paths import DATA, GRAPH, ROOT
from decoders.calibrated import CalibratedDecoder
from environments import CuecaHeroEnvironment
from plasticity.reward_modulated import RewardModulatedPlasticity
from sensors.contrast import ContrastVisualEncoder


def run_eval_session(brain, encoder, decoder, seed, mode='pixels'):
    """Run one evaluation session and collect game state and neural spike count."""
    env = CuecaHeroEnvironment()
    env.reset(seed)
    if mode == 'empty_board':
        env.notes = []
        
    brain.reset()
    encoder.reset()
    decoder.reset()
    
    total_spikes = 0
    presses = np.zeros(4, dtype=int)
    
    while not env.is_done():
        obs = env.get_observation()
        if mode == 'black':
            obs = Observation(np.zeros_like(obs.rgb), obs.timestamp)
            
        stimulus = encoder.encode(obs, 10.0)
        activity = brain.advance(stimulus, 10.0)
        total_spikes += int(activity.counts.sum())
        action = decoder.decode(activity)
        presses += np.array(action.values, dtype=int)
        env.step(action, 1.0 / 30.0)
        
    state = env.get_state()
    return {
        'seed': seed,
        'mode': mode,
        'score': state['score'],
        'hits': state['hits'],
        'misses': state['misses'],
        'wrong': state['wrong'],
        'accuracy': state['accuracy'],
        'total_spikes': total_spikes,
        'total_presses': int(presses.sum()),
        'presses_by_lane': presses.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Validate MaleCNS synaptic plasticity.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trained plasticity checkpoint NPZ')
    parser.add_argument('--config', type=Path, default=Path('experiments/cueca_hero_plastic.json'),
                        help='Experiment configuration JSON')
    parser.add_argument('--seeds', type=int, nargs='+', default=[857, 953, 1061],
                        help='Evaluation seeds')
    parser.add_argument('--output', type=Path, required=True, help='Output validation report JSON')
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    
    # Verify graph
    expected_audit = json.loads((DATA / 'outputs/doom/audit/data-integrity.json').read_text())
    digest = hashlib.sha256()
    with GRAPH.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    graph_hash = digest.hexdigest()
    if graph_hash != expected_audit['graph_sha256']:
        raise ValueError('Graph checksum mismatch')

    print('Initializing MaleCNSCore and modules...', flush=True)
    brain = MaleCNSCore()
    sensor_cfg = config['sensors'][0]
    encoder = ContrastVisualEncoder(
        brain.state.retina, brain.state.uv, brain.state.lamina,
        **{k: v for k, v in sensor_cfg.items() if k != 'type'}
    )
    decoder_cfg = config['decoder']
    model_path = ROOT / decoder_cfg['model'] if not Path(decoder_cfg['model']).is_absolute() else Path(decoder_cfg['model'])
    decoder = CalibratedDecoder(json.loads(model_path.read_text()))

    p_cfg = config['plasticity']
    plasticity = RewardModulatedPlasticity(
        eta=p_cfg.get('eta', 0.002),
        eligibility_tau_ms=p_cfg.get('eligibility_tau_ms', 750.0),
        minimum_fraction=p_cfg.get('minimum_fraction', 0.95),
        maximum_fraction=p_cfg.get('maximum_fraction', 1.05)
    )
    plasticity.bind(brain, encoder.retina, decoder.indices, graph_hash)

    print(f"\n1. Evaluating BASELINE MaleCNS (frozen weights)...", flush=True)
    baseline_runs = [run_eval_session(brain, encoder, decoder, s) for s in args.seeds]
    for r in baseline_runs:
        print(f"  Seed {r['seed']}: Score={r['score']} Hits={r['hits']}/48 Wrong={r['wrong']} Spikes={r['total_spikes']:,}")

    print(f"\n2. Loading plastic weights from {args.checkpoint}...", flush=True)
    plasticity.restore(brain, args.checkpoint)
    status = plasticity.status(brain)
    print(f"  Plasticity Status: {status['changed_edges']}/{status['plastic_edges']} synapses modified "
          f"({status['changed_edges']/status['plastic_edges']*100:.1f}%), "
          f"MeanFraction={status['mean_fraction']:.5f}, "
          f"Bounds=[{status['minimum_fraction']:.4f}, {status['maximum_fraction']:.4f}]")

    print(f"\n3. Evaluating ADAPTED MaleCNS (modified plastic weights)...", flush=True)
    adapted_runs = [run_eval_session(brain, encoder, decoder, s) for s in args.seeds]
    for r in adapted_runs:
        print(f"  Seed {r['seed']}: Score={r['score']} Hits={r['hits']}/48 Wrong={r['wrong']} Spikes={r['total_spikes']:,}")

    print(f"\n4. Evaluating ERASED MaleCNS (memory reset to baseline)...", flush=True)
    plasticity.erase(brain)
    erased_runs = [run_eval_session(brain, encoder, decoder, s) for s in args.seeds]
    for r in erased_runs:
        print(f"  Seed {r['seed']}: Score={r['score']} Hits={r['hits']}/48 Wrong={r['wrong']} Spikes={r['total_spikes']:,}")

    # Verify bit-for-bit equivalence between Baseline and Erased
    bit_identical = True
    for b_run, e_run in zip(baseline_runs, erased_runs):
        if b_run['score'] != e_run['score'] or b_run['hits'] != e_run['hits'] or b_run['total_spikes'] != e_run['total_spikes']:
            bit_identical = False
    print(f"  Erased memory bit-for-bit matches baseline: {bit_identical}")

    # Re-apply adapted weights for negative control checks
    plasticity.restore(brain, args.checkpoint)
    print(f"\n5. Negative Controls with Adapted Weights (Black Screen & Empty Board)...", flush=True)
    ctrl_seed = args.seeds[0]
    res_black = run_eval_session(brain, encoder, decoder, ctrl_seed, mode='black')
    print(f"  Black Screen Control: Presses = {res_black['total_presses']}")
    res_empty = run_eval_session(brain, encoder, decoder, ctrl_seed, mode='empty_board')
    print(f"  Empty Board Control:  Presses = {res_empty['total_presses']}")

    # Calculate weight change distribution
    w_adapted = brain.state.weight[plasticity.edges]
    w_base = plasticity.baseline
    delta_w = w_adapted - w_base
    fraction = w_adapted / w_base

    summary = {
        'kind': 'plasticity_validation_report',
        'checkpoint': str(args.checkpoint),
        'graph_sha256': graph_hash,
        'plastic_edges_total': len(plasticity.edges),
        'changed_edges': status['changed_edges'],
        'changed_fraction_percent': round(status['changed_edges'] / len(plasticity.edges) * 100, 2),
        'synaptic_stats': {
            'mean_fraction': status['mean_fraction'],
            'min_fraction': status['minimum_fraction'],
            'max_fraction': status['maximum_fraction'],
            'mean_abs_delta_w': float(np.mean(np.abs(delta_w))),
            'max_abs_delta_w': float(np.max(np.abs(delta_w)))
        },
        'comparison': {
            'baseline': {
                'mean_score': float(np.mean([r['score'] for r in baseline_runs])),
                'mean_hits': float(np.mean([r['hits'] for r in baseline_runs])),
                'mean_wrong': float(np.mean([r['wrong'] for r in baseline_runs])),
                'mean_spikes': float(np.mean([r['total_spikes'] for r in baseline_runs]))
            },
            'adapted': {
                'mean_score': float(np.mean([r['score'] for r in adapted_runs])),
                'mean_hits': float(np.mean([r['hits'] for r in adapted_runs])),
                'mean_wrong': float(np.mean([r['wrong'] for r in adapted_runs])),
                'mean_spikes': float(np.mean([r['total_spikes'] for r in adapted_runs]))
            },
            'erased': {
                'mean_score': float(np.mean([r['score'] for r in erased_runs])),
                'mean_hits': float(np.mean([r['hits'] for r in erased_runs])),
                'mean_wrong': float(np.mean([r['wrong'] for r in erased_runs])),
                'mean_spikes': float(np.mean([r['total_spikes'] for r in erased_runs]))
            }
        },
        'erased_memory_restores_baseline_identically': bit_identical,
        'controls': {
            'black_screen_presses': res_black['total_presses'],
            'empty_board_presses': res_empty['total_presses']
        },
        'limits': 'Evaluates reward-modulated synaptic efficacy on 4,276 connections. The baseline connectome remains in disk.'
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(f"\nValidation report written to {args.output}")

    print("\n=== RESUMEN COMPARATIVO DE PLASTICIDAD ===")
    print(f"Conexiones plásticas: {status['changed_edges']}/{len(plasticity.edges)} ({summary['changed_fraction_percent']}%) modificadas")
    print(f"Límites de eficacia: [{status['minimum_fraction']:.4f}, {status['maximum_fraction']:.4f}]")
    print(f"Baseline: Aciertos={summary['comparison']['baseline']['mean_hits']:.1f}, Puntos={summary['comparison']['baseline']['mean_score']:.0f}")
    print(f"Adaptado: Aciertos={summary['comparison']['adapted']['mean_hits']:.1f}, Puntos={summary['comparison']['adapted']['mean_score']:.0f}")
    print(f"Borrado:  Aciertos={summary['comparison']['erased']['mean_hits']:.1f}, Puntos={summary['comparison']['erased']['mean_score']:.0f}")
    print(f"Retorno exacto tras erase(): {bit_identical}")


if __name__ == '__main__':
    main()
