"""Training history and checkpoint comparison manager."""
import json
from pathlib import Path
from typing import Any, Mapping
from core.paths import ROOT

TRAINING_DIR = ROOT / 'outputs/training'
PLASTICITY_DIR = ROOT / 'outputs/training_plasticity'


def list_training_runs() -> list[dict[str, Any]]:
    """Scan and list all recorded training runs with metadata."""
    runs = []

    # 1. Check RL runs in outputs/training
    if TRAINING_DIR.exists():
        for path in sorted(TRAINING_DIR.iterdir()):
            if not path.is_dir():
                continue
            run_id = path.name
            curve_file = path / 'generation_curve.json'
            progress_file = path / 'progress.json'
            checkpoints = list(path.glob('*.npz'))

            if curve_file.exists():
                data = json.loads(curve_file.read_text())
                runs.append({
                    'id': f'rl/{run_id}',
                    'name': f'RL · {run_id.upper()}',
                    'type': 'rl_generational',
                    'path': str(path.relative_to(ROOT)),
                    'checkpoints_count': len(checkpoints),
                    'summary': data.get('summary', {}),
                    'eval_seeds': data.get('eval_seeds', []),
                })
            elif progress_file.exists():
                data = json.loads(progress_file.read_text())
                completed = data.get('episodes_completed', 0) if isinstance(data, dict) else len(data)
                kind = data.get('kind', 'rl_progress') if isinstance(data, dict) else 'rl_progress'
                runs.append({
                    'id': f'rl/{run_id}',
                    'name': f'RL · {run_id}',
                    'type': kind,
                    'path': str(path.relative_to(ROOT)),
                    'checkpoints_count': len(checkpoints),
                    'summary': {
                        'episodes': completed,
                    },
                })

    # 2. Check Plasticity runs in outputs/training_plasticity
    if PLASTICITY_DIR.exists():
        for path in sorted(PLASTICITY_DIR.iterdir()):
            if not path.is_dir():
                continue
            run_id = path.name
            progress_file = path / 'progress.json'
            val_file = path / 'validation.json'
            checkpoints = list(path.glob('*.npz'))

            val_summary = {}
            if val_file.exists():
                val_summary = json.loads(val_file.read_text())

            runs.append({
                'id': f'plasticity/{run_id}',
                'name': f'Plasticidad · {run_id.upper()}',
                'type': 'synaptic_plasticity',
                'path': str(path.relative_to(ROOT)),
                'checkpoints_count': len(checkpoints),
                'summary': val_summary,
            })

    return runs


def get_run_curve(run_id: str) -> dict[str, Any]:
    """Retrieve detailed generational curve and checkpoints for a given run."""
    clean_id = run_id.replace('..', '').strip('/')
    if clean_id.startswith('rl/'):
        folder = TRAINING_DIR / clean_id[3:]
    elif clean_id.startswith('plasticity/'):
        folder = PLASTICITY_DIR / clean_id[11:]
    else:
        folder = TRAINING_DIR / clean_id

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f'Run {run_id} not found')

    curve_file = folder / 'generation_curve.json'
    if curve_file.exists():
        data = json.loads(curve_file.read_text())
        # Attach file sizes for checkpoints
        for item in data.get('curve', []):
            ckpt_name = item.get('checkpoint')
            if ckpt_name and ckpt_name != 'baseline':
                ckpt_path = folder / ckpt_name
                if ckpt_path.exists():
                    item['file_size_kb'] = round(ckpt_path.stat().st_size / 1024, 1)
        return data

    progress_file = folder / 'progress.json'
    if progress_file.exists():
        raw_progress = json.loads(progress_file.read_text())
        if isinstance(raw_progress, dict) and 'episodes' in raw_progress:
            ep_list = raw_progress['episodes']
        elif isinstance(raw_progress, list):
            ep_list = raw_progress
        else:
            ep_list = []

        curve = []
        supervised = isinstance(raw_progress, dict) and raw_progress.get('kind') == 'supervised_neural_readout'
        evaluation_file = folder / 'evaluation.json'
        evaluation = json.loads(evaluation_file.read_text()) if evaluation_file.exists() else {}
        final_metrics = evaluation.get('metrics', {})
        for i, p in enumerate(ep_list):
            is_final_evaluation = supervised and i == len(ep_list) - 1 and bool(final_metrics)
            curve.append({
                'generation': p.get('index', p.get('episode', i + 1)),
                'checkpoint': f"{p.get('name', f'episode-{i+1:04d}')}.npz",
                'mean_score': final_metrics.get('score') if is_final_evaluation else p.get('score', p.get('reward')),
                'mean_hits': final_metrics.get('hits') if is_final_evaluation else p.get('hits'),
                'mean_wrong': final_metrics.get('wrong') if is_final_evaluation else p.get('wrong'),
                'mean_misses': final_metrics.get('misses') if is_final_evaluation else p.get('misses'),
                'mean_accuracy': final_metrics.get('accuracy') if is_final_evaluation else p.get('accuracy'),
                'loss': p.get('loss'),
                'changed_edges': p.get('changed_edges', 0),
            })
            checkpoint_path = folder / curve[-1]['checkpoint']
            if checkpoint_path.exists():
                curve[-1]['file_size_kb'] = round(checkpoint_path.stat().st_size / 1024, 1)
            elif supervised:
                curve[-1]['checkpoint'] = None
        return {
            'kind': raw_progress.get('kind', 'progress_curve') if isinstance(raw_progress, dict) else 'progress_curve',
            'run_dir': str(folder.relative_to(ROOT)),
            'summary': {
                'total_steps': len(curve),
                'final_score': final_metrics.get('score'),
                'final_accuracy': final_metrics.get('accuracy'),
            },
            'curve': curve,
        }

    raise ValueError(f'No curve data found in run {run_id}')
