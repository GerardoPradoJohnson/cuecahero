"""Evaluate policy learning progression generation by generation across checkpoints.

Measures accuracy, hits, misses, wrong presses, score and reward on held-out test charts
at every generation/checkpoint to verify consistent learning and policy improvement.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.contracts import Action
from environments import CuecaHeroEnvironment
from training.rl import ReinforcementReadoutTrainer, temporal_features


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


def cache_eval_seeds(seeds, model, cache_dir):
    """Extract and cache neural features for evaluation seeds once."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [s for s in seeds if not (cache_dir / f'eval-{s}.npz').exists()]
    if not missing:
        return

    print(f"Pre-extracting neural features for test seeds {missing} using MaleCNS...", flush=True)
    from core.malecns import MaleCNSCore
    from sensors.contrast import ContrastVisualEncoder
    
    brain = MaleCNSCore()
    encoder = ContrastVisualEncoder(
        brain.state.retina, brain.state.uv, brain.state.lamina,
        **model['sensor']
    )
    groups = [np.asarray(g) for g in model['groups']]
    
    for s in missing:
        t0 = time.perf_counter()
        env = CuecaHeroEnvironment()
        env.reset(s)
        brain.reset()
        encoder.reset()
        filtered = np.zeros(len(groups), dtype=np.float64)
        features = []
        
        while not env.is_done():
            stimulus = encoder.encode(env.get_observation(), 10.0)
            activity = brain.advance(stimulus, 10.0)
            alpha = 1.0 - np.exp(-10.0 / model['tau_ms'])
            rates = np.array([activity.counts[g].mean() for g in groups]) * 100.0
            filtered += alpha * (rates - filtered)
            features.append(filtered.copy())
            env.step(Action(), 1.0 / 30.0)
            
        np.savez_compressed(cache_dir / f'eval-{s}.npz', features=np.asarray(features, dtype=np.float32))
        print(f"Cached seed {s} ({len(features)} frames in {time.perf_counter() - t0:.1f}s)", flush=True)


def evaluate_checkpoint_deterministic(trainer, seed, raw_features):
    """Run one deterministic evaluation episode without exploration noise."""
    env = CuecaHeroEnvironment()
    env.reset(seed)
    trainer.reset_runtime()
    
    X = temporal_features_batch(raw_features, trainer.lags)
    Z = (X - trainer.mean) / trainer.scale
    
    for t in range(len(Z)):
        norm_x = Z[t]
        current_time_ms = env.time * 1000.0
        # explore=False -> strictly deterministic fire if rate >= threshold & armed & cooldown_ok
        action, _ = trainer.step(norm_x, current_time_ms, explore=False)
        env.step(action, 1.0 / 30.0)
        
    state = env.get_state()
    return state


def main():
    parser = argparse.ArgumentParser(description='Evaluate generation-by-generation learning curve.')
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Directory containing training checkpoints (episode-XXXX.npz)')
    parser.add_argument('--base-model', type=Path, default=Path('config/perception-readout.json'),
                        help='Initial model JSON')
    parser.add_argument('--seeds', type=int, nargs='+', default=[857, 953, 1061],
                        help='Evaluation seeds')
    parser.add_argument('--cache-dir', type=Path, default=Path('outputs/calibration/eval-cache'),
                        help='Directory to store/read cached evaluation features')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output JSON file for learning curve report')
    args = parser.parse_args()

    model_data = json.loads(args.base_model.read_text())
    cache_eval_seeds(args.seeds, model_data, args.cache_dir)
    
    # Load cached features for eval seeds
    eval_features = {}
    for s in args.seeds:
        with np.load(args.cache_dir / f'eval-{s}.npz', allow_pickle=False) as d:
            eval_features[s] = d['features']

    # Find checkpoints in run-dir
    ckpts = sorted(args.run_dir.glob('episode-*.npz'))
    if not ckpts:
        parser.error(f"No checkpoint files found in {args.run_dir}")

    curve = []
    
    # Generation 0: Base Model
    trainer_gen0 = ReinforcementReadoutTrainer(model_data)
    gen0_runs = [evaluate_checkpoint_deterministic(trainer_gen0, s, eval_features[s]) for s in args.seeds]
    gen0_rec = {
        'generation': 0,
        'checkpoint': 'baseline',
        'mean_score': float(np.mean([r['score'] for r in gen0_runs])),
        'mean_hits': float(np.mean([r['hits'] for r in gen0_runs])),
        'mean_wrong': float(np.mean([r['wrong'] for r in gen0_runs])),
        'mean_misses': float(np.mean([r['misses'] for r in gen0_runs])),
        'mean_accuracy': float(np.mean([r['accuracy'] or 0.0 for r in gen0_runs])),
        'per_seed': {s: {'score': r['score'], 'hits': r['hits'], 'wrong': r['wrong'], 'accuracy': r['accuracy']}
                     for s, r in zip(args.seeds, gen0_runs)}
    }
    curve.append(gen0_rec)

    # Evaluate each checkpoint (Generation 1 .. N)
    for idx, ckpt_path in enumerate(ckpts):
        gen_num = idx + 1
        trainer = ReinforcementReadoutTrainer.load(ckpt_path)
        # Use threshold from trained model
        runs = [evaluate_checkpoint_deterministic(trainer, s, eval_features[s]) for s in args.seeds]
        rec = {
            'generation': gen_num,
            'checkpoint': ckpt_path.name,
            'mean_score': float(np.mean([r['score'] for r in runs])),
            'mean_hits': float(np.mean([r['hits'] for r in runs])),
            'mean_wrong': float(np.mean([r['wrong'] for r in runs])),
            'mean_misses': float(np.mean([r['misses'] for r in runs])),
            'mean_accuracy': float(np.mean([r['accuracy'] or 0.0 for r in runs])),
            'per_seed': {s: {'score': r['score'], 'hits': r['hits'], 'wrong': r['wrong'], 'accuracy': r['accuracy']}
                         for s, r in zip(args.seeds, runs)}
        }
        curve.append(rec)

    # Print markdown table
    print("\n=== EVALUACIÓN GENERACIÓN TRAS GENERACIÓN (TEST SET: SEEDS " + str(args.seeds) + ") ===")
    print("| Generación | Checkpoint | Aciertos / 48 | Pulsaciones Vacías | Precisión (%) | Puntuación Media |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in curve:
        name = "Gen 0 (Base)" if row['generation'] == 0 else f"Gen {row['generation']}"
        print(f"| {name} | `{row['checkpoint']}` | {row['mean_hits']:.1f} | {row['mean_wrong']:.1f} | **{row['mean_accuracy']:.1f}%** | {row['mean_score']:.0f} |")

    initial_acc = curve[0]['mean_accuracy']
    final_acc = curve[-1]['mean_accuracy']
    max_acc = max(r['mean_accuracy'] for r in curve)
    initial_wrong = curve[0]['mean_wrong']
    final_wrong = curve[-1]['mean_wrong']

    print(f"\nResumen de Evolución:")
    print(f"  Precisión inicial (Gen 0): {initial_acc:.1f}% -> Precisión final (Gen {curve[-1]['generation']}): {final_acc:.1f}% (Máximo: {max_acc:.1f}%)")
    print(f"  Pulsaciones vacías (errores): {initial_wrong:.1f} -> {final_wrong:.1f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            'kind': 'generation_learning_curve_report',
            'run_dir': str(args.run_dir),
            'eval_seeds': args.seeds,
            'summary': {
                'initial_accuracy': initial_acc,
                'final_accuracy': final_acc,
                'max_accuracy': max_acc,
                'accuracy_gain': round(final_acc - initial_acc, 2),
                'initial_wrong': initial_wrong,
                'final_wrong': final_wrong,
                'wrong_reduction': round(initial_wrong - final_wrong, 2),
                'score_gain': round(curve[-1]['mean_score'] - curve[0]['mean_score'], 1)
            },
            'curve': curve
        }
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f"Reporte guardado en {args.output}")


if __name__ == '__main__':
    main()
