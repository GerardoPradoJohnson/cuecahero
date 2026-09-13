"""Controlled visual assay on the REAL complete MaleCNS graph.

Only the test images are constructed. No neural records, edges or activity are
fabricated. Never changes live sessions or upstream sources.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import ROOT, DATA, GRAPH
from core.malecns import MaleCNSCore
from core.contracts import Observation
from environments import CuecaHeroEnvironment
from sensors import VisualEncoder
from decoders import DirectionalDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration-ms', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/diagnostics/visual-readouts.json')
    args = parser.parse_args()
    if args.duration_ms < 10 or args.duration_ms % 10:
        parser.error('duration must be a positive multiple of 10 ms')
    if args.output.exists():
        parser.error('Output exists; choose a new path to preserve prior evidence')
    config = json.loads((ROOT/'experiments/cueca_hero.json').read_text())
    manifest = json.loads((DATA/'outputs/doom/malecns_v1/manifest.json').read_text())
    audit = json.loads((DATA/'outputs/doom/audit/data-integrity.json').read_text())
    digest = hashlib.sha256()
    with GRAPH.open('rb') as stream:
        for block in iter(lambda:stream.read(8 * 1024**2), b''):
            digest.update(block)
    if not audit['passed'] or digest.hexdigest() != audit['graph_sha256']:
        raise RuntimeError('Graph does not match the completed audit')
    brain = MaleCNSCore(backend=config['brain']['backend'])
    encoder = VisualEncoder(brain.state.retina, brain.state.uv, brain.state.lamina,
                            config['sensors'][0]['lamina_bias'])
    groups = [[r['index'] for r in manifest['readouts'] if r['type']==spec['type'] and r['side']==spec['side']]
              for spec in config['decoder']['groups']]
    decoder = DirectionalDecoder(groups, config['decoder']['threshold_hz'], config['decoder']['cooldown_ms'])
    # Same background, note size, height and colors as the real renderer. These
    # are named calibration stimuli, not gameplay records or anatomical data.
    env = CuecaHeroEnvironment()
    env.notes = []
    images = [('empty_board', env.get_observation().rgb.copy())]
    for lane in range(4):
        env.notes = [{'id':0,'lane':lane,'at':.55,'judgement':None}]
        images.append((f'isolated_note_{"DFJK"[lane]}',env.get_observation().rgb.copy()))
    images.append(('black_image', np.zeros_like(images[0][1])))
    results = []
    start = time.perf_counter()
    for name, rgb in images:
        brain.reset(); encoder.reset(); decoder.reset()
        total = np.zeros(brain.n, dtype=np.int64)
        presses = np.zeros(4, dtype=int)
        traces = []
        for step in range(args.duration_ms // 10):
            stimulus = encoder.encode(Observation(rgb, step / 100),10)
            activity = brain.advance(stimulus,10)
            total += activity.counts
            actions = decoder.decode(activity)
            presses += np.asarray(actions.values, dtype=int)
            traces.append([int(activity.counts[g].sum()) for g in groups])
        result = {'stimulus':name,'neural_ms':args.duration_ms,'total_spikes':int(total.sum()),
                  'spiking_neurons':int((total>0).sum()),'readout_spikes_DFJK':[int(total[g].sum()) for g in groups],
                  'presses_DFJK':presses.tolist(),'readout_spikes_per_10ms':traces}
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='readout_spikes_per_10ms'}),flush=True)
    report = {'assay':'cold-start static visual readout diagnostic', 'backend':brain.backend,
              'neurons':brain.n,'edges':len(brain.state.post),'graph_sha256':digest.hexdigest(),'config':config,
              'source_hashes':manifest['source_hashes'],'conditions':results,
              'wall_seconds':time.perf_counter()-start,
              'limits':['Same exact full graph; independent reset for each stimulus.',
                        'Black image retains the declared tonic lamina current; not zero injected current.',
                        'Static artificial calibration images, not recorded gameplay.',
                        'Sensitivity differences do not establish useful perception or learning.',
                        'No change to decoder, weights, or current live session.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Saved {args.output}',flush=True)
if __name__=='__main__':
    main()
