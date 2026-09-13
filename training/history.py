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
                runs.append({
                    'id': f'rl/{run_id}',
                    'name': f'RL · {run_id}',
                    'type': 'rl_progress',
                    'path': str(path.relative_to(ROOT)),
                    'checkpoints_count': len(checkpoints),
                    'summary': {
                        'episodes': len(data) if isinstance(data, list) else 0,
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
        for i, p in enumerate(ep_list):
            curve.append({
                'generation': p.get('index', p.get('episode', i + 1)),
                'checkpoint': f"{p.get('name', f'episode-{i+1:04d}')}.npz",
                'mean_score': p.get('score', p.get('reward', 0)),
                'mean_hits': p.get('hits', 0),
                'mean_wrong': p.get('wrong', 0),
                'mean_misses': p.get('misses', 0),
                'mean_accuracy': p.get('accuracy', 0.0),
                'changed_edges': p.get('changed_edges', 0),
            })
        return {
            'kind': 'progress_curve',
            'run_dir': str(folder.relative_to(ROOT)),
            'summary': {
                'total_steps': len(curve),
                'final_score': curve[-1]['mean_score'] if curve else 0,
            },
            'curve': curve,
        }

    raise ValueError(f'No curve data found in run {run_id}')
