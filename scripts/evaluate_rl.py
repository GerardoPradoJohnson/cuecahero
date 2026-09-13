"""Rigorous closed-loop evaluation of trained RL policy vs baselines.

Evaluates:
  1. Random Action Baseline (chance level with matched firing rate)
  2. Frozen Initial Readout Baseline (before RL training)
  3. Trained RL Readout Model
  4. Negative Controls: Black Screen and Empty Board (must produce 0 presses)

Operates on independent unseen evaluation seeds in closed-loop MaleCNS simulation.
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
from decoders.calibrated import CalibratedDecoder
from environments import CuecaHeroEnvironment
from sensors.contrast import ContrastVisualEncoder


def evaluate_agent(brain, encoder, decoder, seed, mode='pixels', is_random=False, rng=None):
    """Run one evaluation episode and return state and lane statistics."""
    env = CuecaHeroEnvironment()
    env.reset(seed)
    
    if mode == 'empty_board':
        env.notes = []
        
    brain.reset()
    encoder.reset()
    if decoder is not None:
        decoder.reset()
        
    presses = np.zeros(4, dtype=int)
    last_random_press = np.full(4, -1e12)
    
    while not env.is_done():
        obs = env.get_observation()
        if mode == 'black':
            obs = Observation(np.zeros_like(obs.rgb), obs.timestamp)
            
        if is_random:
            # Baseline: independent random presses with ~1.5 Hz Poisson process and 120ms cooldown
            t_ms = env.time * 1000.0
            act = [0.0] * 4
            for lane in range(4):
                if t_ms - last_random_press[lane] >= 120.0 and rng.random() < 0.05:
                    act[lane] = 1.0
                    last_random_press[lane] = t_ms
            action = Action(tuple(act))
        else:
            stimulus = encoder.encode(obs, 10.0)
            activity = brain.advance(stimulus, 10.0)
            action = decoder.decode(activity)
            
        presses += np.array(action.values, dtype=int)
        env.step(action, 1.0 / 30.0)
        
    state = env.get_state()
    hits_by_lane = [l['hits'] for l in state['lanes']]
    wrong_by_lane = [l['wrong'] for l in state['lanes']]
    misses_by_lane = [l['misses'] for l in state['lanes']]
    
    return {
        'seed': seed,
        'mode': mode,
        'score': state['score'],
        'hits': state['hits'],
        'misses': state['misses'],
        'wrong': state['wrong'],
        'accuracy': state['accuracy'],
        'total_presses': int(presses.sum()),
        'presses_by_lane': presses.tolist(),
        'hits_by_lane': hits_by_lane,
        'wrong_by_lane': wrong_by_lane,
        'misses_by_lane': misses_by_lane
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL readout vs baselines.')
    parser.add_argument('--model', type=Path, required=True, help='Trained RL model JSON')
    parser.add_argument('--baseline', type=Path, default=Path('config/perception-readout.json'),
                        help='Frozen initial baseline model JSON')
    parser.add_argument('--output', type=Path, required=True, help='Output evaluation report JSON')
    parser.add_argument('--seeds', type=int, nargs='+', default=[857, 953, 1061],
                        help='Independent evaluation seeds')
    args = parser.parse_args()

    trained_model = json.loads(args.model.read_text())
    base_model = json.loads(args.baseline.read_text())

    print('Initializing MaleCNSCore and ContrastVisualEncoder for evaluation...', flush=True)
    brain = MaleCNSCore()
    encoder = ContrastVisualEncoder(
        brain.state.retina, brain.state.uv, brain.state.lamina,
        **trained_model['sensor']
    )
    
    decoder_trained = CalibratedDecoder(trained_model)
    decoder_base = CalibratedDecoder(base_model)
    rng_random = np.random.default_rng(12345)

    results = {
        'random_baseline': [],
        'frozen_baseline': [],
        'trained_rl': [],
        'controls': []
    }

    started = time.perf_counter()
    print(f"\nEvaluating on seeds {args.seeds}...", flush=True)

    # 1. Random Baseline
    print("--- 1. Random Action Baseline ---")
    for s in args.seeds:
        res = evaluate_agent(brain, encoder, None, s, mode='pixels', is_random=True, rng=rng_random)
        results['random_baseline'].append(res)
        print(f"Seed {s}: Score={res['score']} Hits={res['hits']}/48 Wrong={res['wrong']} Acc={res['accuracy']}%")

    # 2. Frozen Base Model Baseline
    print("\n--- 2. Frozen Initial Readout Baseline ---")
    for s in args.seeds:
        res = evaluate_agent(brain, encoder, decoder_base, s, mode='pixels', is_random=False)
        results['frozen_baseline'].append(res)
        print(f"Seed {s}: Score={res['score']} Hits={res['hits']}/48 Wrong={res['wrong']} Acc={res['accuracy']}%")

    # 3. Trained RL Readout Model
    print("\n--- 3. Trained RL Readout Model ---")
    for s in args.seeds:
        res = evaluate_agent(brain, encoder, decoder_trained, s, mode='pixels', is_random=False)
        results['trained_rl'].append(res)
        print(f"Seed {s}: Score={res['score']} Hits={res['hits']}/48 Wrong={res['wrong']} Acc={res['accuracy']}%")

    # 4. Controls on trained RL model (Black Screen and Empty Board)
    print("\n--- 4. Negative Controls (Trained Model) ---")
    ctrl_seed = args.seeds[0]
    res_black = evaluate_agent(brain, encoder, decoder_trained, ctrl_seed, mode='black', is_random=False)
    results['controls'].append(res_black)
    print(f"Control Black Screen (Seed {ctrl_seed}): Total Presses = {res_black['total_presses']}")

    res_empty = evaluate_agent(brain, encoder, decoder_trained, ctrl_seed, mode='empty_board', is_random=False)
    results['controls'].append(res_empty)
    print(f"Control Empty Board (Seed {ctrl_seed}): Total Presses = {res_empty['total_presses']}")

    elapsed = time.perf_counter() - started
    
    # Compute aggregates
    def summarize(runs):
        return {
            'mean_hits': round(float(np.mean([r['hits'] for r in runs])), 1),
            'mean_wrong': round(float(np.mean([r['wrong'] for r in runs])), 1),
            'mean_score': round(float(np.mean([r['score'] for r in runs])), 1),
            'mean_accuracy': round(float(np.mean([r['accuracy'] or 0 for r in runs])), 1)
        }

    summary = {
        'random_baseline': summarize(results['random_baseline']),
        'frozen_baseline': summarize(results['frozen_baseline']),
        'trained_rl': summarize(results['trained_rl']),
        'controls_passed': (res_black['total_presses'] == 0 and res_empty['total_presses'] == 0)
    }

    report = {
        'kind': 'rl_evaluation_report',
        'seeds': args.seeds,
        'wall_seconds': round(elapsed, 2),
        'summary': summary,
        'details': results,
        'limits': 'Evaluation of closed-loop external policy on unseen seeds. MaleCNS connectome weights are frozen.'
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"\nEvaluation complete in {elapsed:.1f}s. Report written to {args.output}")
    print("\n=== COMPARATIVE SUMMARY ===")
    print(f"Random Baseline: Hits={summary['random_baseline']['mean_hits']}/48, Wrong={summary['random_baseline']['mean_wrong']}, Score={summary['random_baseline']['mean_score']}")
    print(f"Frozen Baseline: Hits={summary['frozen_baseline']['mean_hits']}/48, Wrong={summary['frozen_baseline']['mean_wrong']}, Score={summary['frozen_baseline']['mean_score']}")
    print(f"Trained RL:      Hits={summary['trained_rl']['mean_hits']}/48, Wrong={summary['trained_rl']['mean_wrong']}, Score={summary['trained_rl']['mean_score']}")
    print(f"Controls Passed: {summary['controls_passed']} (0 presses on black and empty board)")


if __name__ == '__main__':
    main()
