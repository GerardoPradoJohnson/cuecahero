"""Train/resume a supervised external reader at episode boundaries."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.readout import EpisodicReadoutTrainer


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model',type=Path)
    source.add_argument('--resume',type=Path)
    parser.add_argument('--episodes',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stop-after',type=int,help='Total consumed episodes, including restored episodes')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; use a new directory for each run/resume')
    trainer=EpisodicReadoutTrainer.load(args.resume) if args.resume else EpisodicReadoutTrainer(json.loads(args.model.read_text()))
    declared=trainer.template.get('training_files_sha256',{})
    if not declared:parser.error('Model must declare the training files and checksums')
    paths=[]
    for name,digest in sorted(declared.items()):
        if Path(name).name!=name:parser.error('Training filenames must be basenames')
        path=args.episodes/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:parser.error(f'Changed training episode: {name}')
        paths.append(path)
    consumed=[e['name'] for e in trainer.episodes]
    if consumed!=[p.name for p in paths[:len(consumed)]]:parser.error('Checkpoint episode order does not match the split')
    stop=args.stop_after if args.stop_after is not None else len(paths)
    if not len(consumed)<=stop<=len(paths):parser.error('Invalid stop-after count')
    args.output.mkdir(parents=True)
    trainer.save(args.output/f'episode-{len(consumed):04d}.npz')
    for path in paths[len(consumed):stop]:
        result=trainer.consume(path)
        trainer.save(args.output/f'episode-{len(trainer.episodes):04d}.npz')
        print(json.dumps(result),flush=True)
    (args.output/'model.json').write_text(json.dumps(trainer.model(),indent=2)+'\n')
    (args.output/'progress.json').write_text(json.dumps({
        'kind':'supervised_external_readout','episodes':trainer.episodes,
        'complete':len(trainer.episodes)==len(paths),
        'limits':'Per-episode MSE is training fit, not gameplay performance. No reward learning or connectome plasticity. Checkpoints resume at episode boundaries, not midway through a brain simulation.'},indent=2)+'\n')

if __name__=='__main__':main()
