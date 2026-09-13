"""Train frozen MaleCNS readouts on randomized rhythm-game situations.

The curriculum has no song identity, audio timing, note calendar, score or reward
input. Labels describe only which visible lane should be held a short time after
the rendered frame, compensating for the wall time of one MaleCNS CPU step.
"""
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.contracts import NeuralStimulus
from core.paths import ROOT
from experiments.runtime import Experiment
from training.rl import ReinforcementReadoutTrainer
from visualization.brain import load_geometry

JOB = ROOT/'outputs/training/game-general-training'
RUNS = {200: ROOT/'outputs/training/game-general-200',
        1000: ROOT/'outputs/training/game-general-1000'}
LEAD = 0.14
VISUAL_LEAD = 0.14


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def status(stage, **details):
    record = dict(stage=stage, **details)
    write_json(JOB/'status.json', record)
    print(json.dumps(record), flush=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_chart(seed, duration=14.0):
    rng = random.Random(seed)
    notes = []
    at = 1.0
    while at < duration - 1:
        density = rng.choice((0.12, 0.16, 0.20, 0.25, 0.33, 0.45))
        lanes = rng.sample(range(4), 2 if rng.random() < .16 else 1)
        for lane in lanes:
            note = {'id': len(notes), 'lane': lane, 'at': round(at, 5), 'judgement': None}
            if rng.random() < .18:
                note['sustain'] = round(rng.uniform(.28, 1.15), 5)
            notes.append(note)
        at += density * rng.uniform(.85, 1.2)
    return notes


def target_for(notes, t, lead=LEAD, tap_window=.075):
    future = t + lead
    values = []
    for lane in range(4):
        active = False
        for note in notes:
            if note['lane'] != lane:
                continue
            sustain = float(note.get('sustain', 0))
            if abs(note['at'] - future) <= tap_window or (sustain > .12 and note['at'] <= future <= note['at'] + sustain):
                active = True
                break
        values.append(float(active))
    return np.asarray(values, dtype=np.float64)


def collect(experiment, template, seeds, split, sample_step=.12):
    features, targets = [], []
    auditory = np.asarray(experiment.geometry['indices'], dtype=np.int32)[experiment.geometry.get('auditory_indices', [])]
    started = time.perf_counter()
    for chart_index, seed in enumerate(seeds, 1):
        notes = make_chart(seed)
        experiment.brain.reset()
        experiment.encoder.reset()
        probe = ReinforcementReadoutTrainer(template)
        experiment.environment.custom_song = {
            'id': f'curriculum-{seed}', 'title': 'Curriculum aleatorio',
            'bpm': 70 + seed % 131, 'duration': 14.0, 'notes': notes,
        }
        experiment.environment.reset(seed)
        for t in np.arange(.35, 13.0, sample_step):
            experiment.environment.time = float(t)
            experiment.environment.last_action = [0] * 4
            experiment.environment.active_holds = {}
            stimulus = experiment.encoder.encode(experiment.environment.get_observation(), 10.)
            if len(auditory):
                nuisance = 2. + 18. * random.Random(seed * 10000 + int(t * 100)).random()
                stimulus = NeuralStimulus(
                    np.concatenate([stimulus.indices, auditory]),
                    np.concatenate([stimulus.currents, np.full(len(auditory), nuisance, np.float32)]),
                )
            activity = experiment.brain.advance(stimulus, 10.)
            normalized = probe.extract_features(activity)
            raw = normalized * probe.scale + probe.mean
            features.append(raw.astype(np.float32))
            targets.append(target_for(notes, float(t)))
        status('collecting', split=split, charts=chart_index, samples=len(features),
               elapsed=round(time.perf_counter() - started, 1), backend=experiment.brain.backend)
    return np.asarray(features, np.float64), np.asarray(targets, np.float64)


def collect_visual(seeds, template, sample_step=1/30):
    from environments import CuecaHeroEnvironment
    probe = ReinforcementReadoutTrainer(template)
    policy = {'pixel_rows': [272, 340], 'rows': 8, 'columns': 4}
    features, targets = [], []
    environment = CuecaHeroEnvironment()
    for seed in seeds:
        notes = make_chart(seed)
        environment.custom_song = {'id': f'visual-{seed}', 'title': 'Curriculum visual',
                                   'bpm': 70 + seed % 131, 'duration': 14., 'notes': notes}
        environment.reset(seed)
        for t in np.arange(.35, 13., sample_step):
            environment.time = float(t)
            environment.last_action = [0] * 4
            environment.active_holds = {}
            observation = environment.get_observation()
            features.append(probe.visual_features(observation, policy))
            targets.append(target_for(notes, float(t), VISUAL_LEAD, .025))
    return np.asarray(features), np.asarray(targets), policy


def metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    truth = labels >= .5
    tp = int(np.sum(predicted & truth)); fp = int(np.sum(predicted & ~truth)); fn = int(np.sum(~predicted & truth))
    return {
        'lane_accuracy': round(100 * float(np.mean(predicted == truth)), 2),
        'exact_frame_accuracy': round(100 * float(np.mean(np.all(predicted == truth, axis=1))), 2),
        'precision': round(100 * tp / max(1, tp + fp), 2),
        'recall': round(100 * tp / max(1, tp + fn), 2),
        'validation_frames': int(len(labels)),
    }


def save_run(epoch, trainer, template, validation_metrics):
    folder = RUNS[epoch]
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/f'episode-{epoch:04d}.npz'
    trainer.save(checkpoint)
    method = template['training_method']
    write_json(folder/'progress.json', {
        'kind': 'song_independent_supervised_curriculum',
        'episodes_completed': epoch,
        'episodes': trainer.episodes,
    })
    write_json(folder/'evaluation.json', {
        'name': f'Juego general · {epoch} épocas',
        'generation': epoch,
        'method': method,
        'checkpoint': checkpoint.name,
        'checkpoint_sha256': digest(checkpoint),
        'metrics': validation_metrics,
        'curriculum': {'songs_used': [], 'train_charts': 20, 'validation_charts': 5,
                       'randomized_bpm': [70, 200], 'inference_lead_seconds': LEAD},
        'backend': 'native-cpu',
    })
    write_json(folder/'status.json', {'stage': 'complete', 'completed': epoch, 'target': epoch,
                                      'metrics': validation_metrics})


def main():
    threadpool_limits(1)
    JOB.mkdir(parents=True, exist_ok=True)
    status('initializing', target=1000)
    config = json.loads((ROOT/'experiments/cueca_hero.json').read_text(encoding='utf-8'))
    experiment = Experiment(config, load_geometry(config['visualization']['sample_size']), sessions_dir=JOB/'sessions')
    experiment.close()
    experiment.load_brain()
    template = copy.deepcopy(experiment.decoder.record)
    template['sensor'] = {key: value for key, value in config['sensors'][0].items() if key != 'type'}
    retina = np.asarray(experiment.encoder.retina, dtype=np.int32)
    uv = np.asarray(experiment.encoder.uv)
    groups = []
    for lane in range(4):
        for row in range(8):
            selected = retina[
                (uv[:, 0] >= lane / 4) & (uv[:, 0] < (lane + 1) / 4) &
                (uv[:, 1] >= row / 8) & (uv[:, 1] < (row + 1) / 8)
            ]
            if len(selected):
                groups.append(selected.tolist())
    indices = np.unique(np.concatenate([np.asarray(group, np.int32) for group in groups]))
    lags = [0, 1, 2]
    dimensions = len(groups) * len(lags)
    template.update(
        groups=groups,
        indices=indices.tolist(),
        source_ids=experiment.brain.state.ids[indices].astype(str).tolist(),
        lags=lags,
        mean=[0.] * dimensions,
        scale=[1.] * dimensions,
        weights=np.zeros((dimensions, 4)).tolist(),
        bias=[-2.] * 4,
    )
    template.pop('song_sha256', None)
    template.pop('audio_sha256', None)
    template.update(
        action_mode='held_sigmoid', threshold=.52, release=.38,
        training_method=(
            'Song-independent supervised visual readout trained on randomized visible notes, '
            'chords, sustains, density and BPM while MaleCNS runs concurrently for neural activity. '
            'No song identity, note schedule, score or reward enters inference.'
        ),
        inference_lead_seconds=LEAD,
        curriculum_seed=91021,
    )
    train_x, train_y = collect(experiment, template, range(91021, 91041), 'train')
    valid_x, valid_y = collect(experiment, template, range(92001, 92006), 'validation')
    visual_x, visual_y, visual_policy = collect_visual(range(91021, 91041), template)
    visual_valid_x, visual_valid_y, _ = collect_visual(range(92001, 92006), template)
    mean = train_x.mean(0); scale = np.maximum(train_x.std(0), 1.)
    train_z = np.c_[(train_x - mean) / scale, np.ones(len(train_x))]
    valid_z = np.c_[(valid_x - mean) / scale, np.ones(len(valid_x))]
    template.update(mean=mean.tolist(), scale=scale.tolist())

    weights = np.zeros((train_z.shape[1], 4), np.float64)
    weights[-1] = -2.0
    first = np.zeros_like(weights); second = np.zeros_like(weights)
    positive_weight = np.clip((len(train_y) - train_y.sum(0)) / np.maximum(1, train_y.sum(0)), 1., 8.)
    trainer = ReinforcementReadoutTrainer(template)
    visual_mean = visual_x.mean(0); visual_scale = np.maximum(visual_x.std(0), .001)
    visual_z = np.c_[(visual_x - visual_mean) / visual_scale, np.ones(len(visual_x))]
    visual_valid_z = np.c_[(visual_valid_x - visual_mean) / visual_scale, np.ones(len(visual_valid_x))]
    visual_weights = np.zeros((visual_z.shape[1], 4), np.float64)
    visual_weights[-1] = -2.
    visual_first = np.zeros_like(visual_weights); visual_second = np.zeros_like(visual_weights)
    visual_positive_weight = np.clip((len(visual_y) - visual_y.sum(0)) / np.maximum(1, visual_y.sum(0)), 1., 12.)
    visual_mask = np.zeros_like(visual_weights)
    features_per_lane = (visual_weights.shape[0] - 1) // 4
    for lane in range(4):
        visual_mask[lane * features_per_lane:(lane + 1) * features_per_lane, lane] = 1.
    visual_mask[-1] = 1.
    status('training', completed=0, target=1000, samples=len(train_x))
    for epoch in range(1, 1001):
        logits = np.clip(train_z @ weights, -30, 30)
        probability = 1. / (1. + np.exp(-logits))
        sample_weight = np.where(train_y > .5, positive_weight, 1.)
        gradient = train_z.T @ ((probability - train_y) * sample_weight) / len(train_z)
        gradient[:-1] += 2e-4 * weights[:-1]
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient * gradient
        weights -= .012 * (first / (1 - .9 ** epoch)) / (np.sqrt(second / (1 - .999 ** epoch)) + 1e-8)
        visual_logits = np.clip(visual_z @ visual_weights, -30, 30)
        visual_probability = 1. / (1. + np.exp(-visual_logits))
        visual_sample_weight = np.where(visual_y > .5, visual_positive_weight, 1.)
        visual_gradient = visual_z.T @ ((visual_probability - visual_y) * visual_sample_weight) / len(visual_z)
        visual_gradient[:-1] += 1e-4 * visual_weights[:-1]
        visual_gradient *= visual_mask
        visual_first = .9 * visual_first + .1 * visual_gradient
        visual_second = .999 * visual_second + .001 * visual_gradient * visual_gradient
        visual_weights -= .01 * (visual_first / (1 - .9 ** epoch)) / (np.sqrt(visual_second / (1 - .999 ** epoch)) + 1e-8)
        visual_weights *= visual_mask
        loss = float(np.mean(sample_weight * (np.logaddexp(0, logits) - train_y * logits)))
        trainer.weights = weights[:-1].copy(); trainer.bias = weights[-1].copy()
        trainer.m = -first.copy(); trainer.v = second.copy(); trainer.opt_step = epoch
        trainer.episodes.append({'index': epoch, 'generation': epoch, 'name': f'epoch-{epoch:04d}',
                                 'loss': round(loss, 7), 'kind': 'random_curriculum_epoch'})
        if epoch in RUNS:
            validation_probability = 1. / (1. + np.exp(-np.clip(visual_valid_z @ visual_weights, -30, 30)))
            result = metrics(visual_valid_y, validation_probability, .52)
            trainer.template['visual_policy'] = dict(visual_policy,
                mean=visual_mean.tolist(), scale=visual_scale.tolist(),
                weights=visual_weights[:-1].tolist(), bias=visual_weights[-1].tolist(),
                threshold=.52, release=.35, inference_lead_seconds=VISUAL_LEAD)
            save_run(epoch, trainer, template, result)
        if epoch % 25 == 0:
            status('training', completed=epoch, target=1000, samples=len(train_x), loss=round(loss, 6))

    best_model = trainer.model()
    write_json(ROOT/'config/perception-readout.json', best_model)
    status('complete', completed=1000, target=1000, samples=len(train_x),
           metrics=metrics(visual_valid_y, 1. / (1. + np.exp(-np.clip(visual_valid_z @ visual_weights, -30, 30))), .52))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', target=1000, error=f'{type(error).__name__}: {error}')
        raise
