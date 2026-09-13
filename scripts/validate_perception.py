"""Closed-loop evaluation of a frozen BCI readout on unseen full charts.

Only rendered pixels enter the encoder, and only neural activity enters the
decoder. Chart labels are inspected afterwards for evaluation, never control.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import GRAPH
from core.contracts import Observation
from core.malecns import MaleCNSCore
from decoders.calibrated import CalibratedDecoder
from environments import CuecaHeroEnvironment
from sensors.contrast import ContrastVisualEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 73, 109])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    model_bytes = args.model.read_bytes()
    model = json.loads(model_bytes)
    digest = hashlib.sha256()
    with GRAPH.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != model['graph_sha256']:
        raise ValueError('Calibration graph mismatch')
    brain = MaleCNSCore()
    if brain.state.ids[model['indices']].astype(str).tolist() != model['source_ids']:
        raise ValueError('Calibration neuron IDs mismatch')
    encoder = ContrastVisualEncoder(brain.state.retina, brain.state.uv,
                                   brain.state.lamina, **model['sensor'])
    decoder = CalibratedDecoder(model)
    results = []
    started = time.perf_counter()
    for mode, seed in [('pixels', s) for s in args.seeds] + [('black', args.seeds[0]), ('empty_board', args.seeds[0])]:
        env = CuecaHeroEnvironment()
        env.reset(seed)
        if mode == 'empty_board':
            env.notes = []
        brain.reset()
        encoder.reset()
        decoder.reset()
        presses = np.zeros(4, int)
        while not env.is_done():
            observation = env.get_observation()
            if mode == 'black':
                observation = Observation(np.zeros_like(observation.rgb), observation.timestamp)
            activity = brain.advance(encoder.encode(observation, 10), 10)
            action = decoder.decode(activity)
            presses += np.array(action.values, dtype=int)
            env.step(action, 1 / 30)
        result = {'mode': mode, 'seed': seed, 'state': env.get_state(),
                  'presses_by_lane': presses.tolist(),
                  'hits_by_lane': [sum(n['lane'] == lane and n['judgement'] in ('good', 'perfect') for n in env.notes) for lane in range(4)]}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {'model_sha256': hashlib.sha256(model_bytes).hexdigest(),
              'graph_sha256': digest.hexdigest(), 'results': results,
              'wall_seconds': time.perf_counter() - started,
              'limits': 'Frozen external decoder; no MaleCNS plasticity or biological learning.'}
    report['activation_criteria'] = 'Every chart: at least 24/48 hits, all four lanes hit, at most 10 false presses. Both controls: zero presses.'
    report['activation_passed'] = all(
        (r['state']['hits'] >= 24 and min(r['hits_by_lane']) > 0 and r['state']['wrong'] <= 10)
        if r['mode'] == 'pixels' else sum(r['presses_by_lane']) == 0
        for r in results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
