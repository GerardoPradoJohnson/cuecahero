"""Install a frozen readout only after its matching evaluation passes the gate."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.paths import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    args=parser.parse_args()
    content=args.model.read_bytes();model=json.loads(content)
    report=json.loads(args.evaluation.read_text())
    if report.get('model_sha256')!=hashlib.sha256(content).hexdigest() or report.get('graph_sha256')!=model['graph_sha256']:
        raise ValueError('Evaluation does not match the frozen model')
    games=[r for r in report['results'] if r['mode']=='pixels']
    controls=[r for r in report['results'] if r['mode']!='pixels']
    if len({r['seed'] for r in games})<3 or {r['mode'] for r in controls}!={'black','empty_board'}:
        raise ValueError('Three independent charts and both controls required')
    if not report.get('activation_passed') or not all(r['state']['hits']>=24 and min(r['hits_by_lane'])>0 and r['state']['wrong']<=10 for r in games) or any(sum(r['presses_by_lane']) for r in controls):
        raise ValueError('Readout failed the activation gate; preserve current configuration')
    config_path=ROOT/'experiments/cueca_hero.json'
    config=json.loads(config_path.read_text())
    legacy=ROOT/'experiments/cueca_hero_legacy.json'
    if not legacy.exists():legacy.write_bytes(config_path.read_bytes())
    target=ROOT/'config/perception-readout.json'
    target.write_bytes(content)
    config['name']='Cueca Hero · lector externo calibrado'
    config['sensors']=[dict(type='contrast_visual',**model['sensor'])]
    config['decoder']={'type':'calibrated','model':str(target.relative_to(ROOT))}
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    (ROOT/'docs/validation/perception-evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Frozen readout installed. Restart scripts/serve.py to load it.')

if __name__=='__main__':main()
