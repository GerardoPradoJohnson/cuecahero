"""Composition root and single-owner simulation worker."""
import base64
import copy
import hashlib
import io
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from core.clock import SimulationClock
from core.contracts import Action, NeuralStimulus
from core.paths import ROOT, DATA, GRAPH
from core.graph_identity import calibration_graph_matches
from core.registry import Registry
from environments import CuecaHeroEnvironment, Navigation2DEnvironment
from sensors import VisualEncoder, ContrastVisualEncoder, AuditoryEncoder, ProprioceptiveEncoder, CompositeMultimodalEncoder
from decoders import DirectionalDecoder, ContinuousRateDecoder
from decoders.calibrated import CalibratedDecoder
from embodiments import GameController, ContinuousEmbodiment
from plasticity import NoPlasticity, RewardModulatedPlasticity
from telemetry import Metrics
from replay import SessionRecorder

registry = Registry()
registry.register('environment', 'cueca_hero', CuecaHeroEnvironment)
registry.register('environment', 'navigation_2d', Navigation2DEnvironment)
registry.register('sensor', 'visual', VisualEncoder)
registry.register('sensor', 'contrast_visual', ContrastVisualEncoder)
registry.register('sensor', 'auditory', AuditoryEncoder)
registry.register('sensor', 'proprioception', ProprioceptiveEncoder)
registry.register('sensor', 'composite_multimodal', CompositeMultimodalEncoder)
registry.register('decoder', 'directional', DirectionalDecoder)
registry.register('decoder', 'continuous_rate', ContinuousRateDecoder)
registry.register('decoder', 'calibrated', CalibratedDecoder)
registry.register('embodiment', 'game_controller', GameController)
registry.register('embodiment', 'continuous', ContinuousEmbodiment)
registry.register('plasticity', 'none', NoPlasticity)
registry.register('plasticity', 'reward_modulated', RewardModulatedPlasticity)

class Experiment:
    def __init__(self, config, geometry, sessions_dir=None):
        self.session_directory = sessions_dir or ROOT / "outputs/sessions"
        self.config = copy.deepcopy(config)
        plasticity_config = dict(config['plasticity'])
        plasticity_type = plasticity_config.pop('type')
        enabled = plasticity_config.pop('enabled')
        if enabled != (plasticity_type != 'none'):
            raise ValueError('Plasticity type and enabled flag disagree')
        if config['brain']['dataset'] != 'malecns_v1' or len(config['sensors']) < 1:
            raise ValueError('This composition supports MaleCNS v1.0 and at least one sensor encoder')
        self.geometry = geometry
        env_config = dict(config['environment'])
        self.environment = registry.create('environment', env_config.pop('type'), **env_config)
        embodiment_config = config['embodiment']
        if isinstance(embodiment_config, dict):
            body_kwargs = dict(embodiment_config)
            body_type = body_kwargs.pop('type')
            self.body = registry.create('embodiment', body_type, **body_kwargs)
        else:
            self.body = registry.create('embodiment', embodiment_config)
        self.plasticity = registry.create('plasticity', plasticity_type, **plasticity_config)
        self.clock = SimulationClock(**config['clock'])
        self.brain = self.encoder = self.decoder = None
        self.driver = 'manual'
        self.mode = 'paused'
        self.recorder = None
        saved = list(self.session_directory.glob('*.jsonl'))
        self.last_session = max(saved, key=lambda p:p.stat().st_mtime).stem if saved else None
        self.last_memory = None
        self.error = None
        self.provenance = {'doomfly':json.loads((ROOT/'config/doomfly.lock.json').read_text()),
                           'graph_sha256':None, 'weights':'plastic' if enabled else 'fixed', 'dataset':'MaleCNS v1.0'}
        self.commands = queue.Queue(maxsize=128)
        self.shutdown_event = threading.Event()
        self.pending = set()
        self.held_keys = set()
        self.revision = 0
        self.metrics = Metrics()
        self.live_training_enabled = False
        self.trainer = None
        self.generation = 0
        self.training_history = []
        self.current_trajectory = []
        self.current_rewards_by_lane = []
        self.realtime_neural = bool(config['brain'].get('realtime', False))
        self.neural_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='flylab-neural')
        self.neural_future = None
        self.latest_neural_activity = None
        self.latest_neural_action = Action()
        self.next_neural_submit = 0.
        self.last_realtime_wall = None
        self.environment.reset(config['seed'])
        self.snapshot = {}
        self.publish()
        self.thread = threading.Thread(target=self.loop, name='flylab-simulation', daemon=True)
        self.thread.start()

    def load_brain(self):
        from core.malecns import MaleCNSCore
        expected = json.loads((DATA/'outputs/doom/audit/data-integrity.json').read_text())
        digest = hashlib.sha256()
        with GRAPH.open('rb') as stream:
            for block in iter(lambda:stream.read(8 * 1024**2), b''):
                digest.update(block)
        if not expected['passed'] or digest.hexdigest() != expected['graph_sha256']:
            raise ValueError('Runtime graph does not match the completed data audit; run prepare_data.py --audit')
        self.provenance['graph_sha256'] = digest.hexdigest()
        brain = MaleCNSCore(backend=self.config['brain']['backend'])
        encoders = []
        for sensor_cfg in self.config['sensors']:
            sensor = dict(sensor_cfg)
            sensor_type = sensor.pop('type')
            if sensor_type in ('visual', 'contrast_visual'):
                enc = registry.create('sensor', sensor_type, retina=brain.state.retina,
                                      uv=brain.state.uv, lamina=brain.state.lamina, **sensor)
            else:
                enc = registry.create('sensor', sensor_type, **sensor)
            encoders.append(enc)
        encoder = encoders[0] if len(encoders) == 1 else CompositeMultimodalEncoder(encoders)
        decoder = dict(self.config['decoder'])
        if decoder['type'] == 'calibrated':
            model_path = ROOT / decoder['model']
            model = json.loads(model_path.read_text())
            if not calibration_graph_matches(GRAPH, model['graph_sha256'], digest.hexdigest()):
                raise ValueError('Calibration does not match graph content')
            if len(encoders) != 1 or model['sensor'] != sensor:
                raise ValueError('Calibration does not match sensor parameters')
            if self.clock.game_step != 1/30 or self.clock.neural_step_ms != 10:
                raise ValueError('This readout calibration requires 1/30 s game and 10 ms neural steps')
            decoder_instance = registry.create('decoder', 'calibrated', model=model)
            if not (brain.state.ids[model['indices']].astype(str) == model['source_ids']).all():
                raise ValueError('Calibration source neuron IDs do not match the graph')
            self.provenance['decoder_sha256'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
            from training.rl import ReinforcementReadoutTrainer
            self.trainer = ReinforcementReadoutTrainer(model, lr=3e-3, explore_temp=0.35)
            if self.live_training_enabled:
                self.init_trainer_from_scratch()
        else:
            self.trainer = None
            specifications = decoder.pop('groups')
            readouts = json.loads((DATA/'outputs/doom/malecns_v1/manifest.json').read_text())['readouts']
            groups = [[r['index'] for r in readouts if r['type'] == spec['type'] and r['side'] == spec['side']] for spec in specifications]
            decoder_instance = registry.create('decoder', decoder.pop('type'), groups=groups, **decoder)
        if self.plasticity.enabled:
            if self.config['decoder']['type'] != 'calibrated':
                raise ValueError('Reward plasticity requires the calibrated visual readout')
            self.plasticity.bind(brain, encoder.retina, decoder_instance.indices, digest.hexdigest())
            self.provenance['plasticity'] = self.plasticity.status(brain)
        self.brain, self.encoder, self.decoder = brain, encoder, decoder_instance

    def init_trainer_from_scratch(self):
        if not self.trainer:
            return
        import numpy as np
        self.trainer.weights = np.zeros_like(self.trainer.weights)
        self.trainer.bias = np.full(4, -0.6, dtype=np.float64)
        self.trainer.m.fill(0)
        self.trainer.v.fill(0)
        self.trainer.opt_step = 0
        self.trainer.baseline_return.fill(0)
        self.trainer.baseline_count = 0
        self.trainer.reset_runtime()
        self.generation = 0
        self.training_history = []
        self.current_trajectory = []
        self.current_rewards_by_lane = []

    def reset(self):
        self._drain_neural_job()
        if self.recorder:
            self.recorder.close()
            self.recorder = None
        self.environment.reset(self.config['seed'])
        self.clock.reset()
        self.metrics = Metrics()
        self.pending.clear()
        self.held_keys.clear()
        if self.brain:
            self.brain.reset()
            self.encoder.reset()
            self.decoder.reset()
        if self.trainer:
            self.trainer.reset_runtime()
            self.current_trajectory = []
            self.current_rewards_by_lane = []
        self.mode = 'paused'
        self.error = None

    def _drain_neural_job(self):
        if self.neural_future is not None:
            try:
                self.neural_future.result(timeout=10)
            finally:
                self.neural_future = None
        self.latest_neural_activity = None
        self.latest_neural_action = Action()
        self.next_neural_submit = 0.
        self.last_realtime_wall = None

    def _neural_step(self, observation, acoustic_energy, game_time):
        import numpy as np
        stimulus = self.encoder.encode(observation, self.clock.neural_step_ms)
        aud_local = self.geometry.get('auditory_indices', [])
        if len(aud_local) > 0:
            aud_indices = np.asarray(self.geometry['indices'], dtype=np.int32)[aud_local]
            aud_currents = np.full(len(aud_indices), 18.0 * acoustic_energy + 2.0, dtype=np.float32)
            stimulus = NeuralStimulus(
                indices=np.concatenate([stimulus.indices, aud_indices]),
                currents=np.concatenate([stimulus.currents, aud_currents])
            )
        activity = self.brain.advance(stimulus, self.clock.neural_step_ms)
        if self.trainer is not None:
            if self.trainer.template.get('visual_policy'):
                return activity, None
            features = self.trainer.extract_features(activity)
            action, _ = self.trainer.step(features, game_time * 1000.0, explore=False)
        else:
            action = self.decoder.decode(activity)
        return activity, action

    def _tick_realtime_neural(self, started):
        now = time.perf_counter()
        elapsed = self.clock.game_step if self.last_realtime_wall is None else now - self.last_realtime_wall
        self.last_realtime_wall = now
        game_dt = min(.2, max(.001, elapsed * self.clock.speed))
        activity = None
        if self.neural_future is not None and self.neural_future.done():
            activity, action = self.neural_future.result()
            if action is not None:
                self.latest_neural_action = action
            self.latest_neural_activity = activity
            self.neural_future = None

        if self.trainer.template.get('visual_policy'):
            self.latest_neural_action = self.trainer.visual_step(self.environment.get_observation())
        self.environment.step(self.body.apply(self.latest_neural_action), game_dt)
        if (self.neural_future is None and not self.environment.is_done() and
                now >= self.next_neural_submit):
            self.neural_future = self.neural_executor.submit(
                self._neural_step,
                self.environment.get_observation(),
                self.acoustic_energy,
                self.environment.time,
            )
            self.next_neural_submit = now + .25

        self.clock.advance(neural=activity is not None)
        telemetry = self.metrics.record(activity, time.perf_counter() - started,
                                        self.environment.get_observation().timestamp)
        if self.environment.is_done():
            self.mode = 'finished'
        self.publish(self.latest_neural_activity, telemetry)
        self._record_frame()

    def _record_frame(self):
        if self.recorder:
            self.recorder.write({'kind':'frame', 'snapshot':self.snapshot})
            if self.mode == 'finished':
                self.recorder.close()
                self.recorder = None
                if self.plasticity.enabled:
                    path = ROOT/'outputs/memory'/(self.last_session+'.npz')
                    self.plasticity.checkpoint(self.brain, path)
                    self.last_memory = str(path.relative_to(ROOT))

    def enqueue(self, command):
        self.commands.put_nowait(command)

    def control(self, command):
        kind = command['type']
        if kind == 'driver':
            target = command['value']
            self.reset()
            if target == 'neural' and self.brain is None:
                self.mode = 'loading'
                self.publish()
                self.load_brain()
            self.driver = target
            self.mode = 'paused'
        elif kind == 'start':
            if self.environment.is_done():
                self.reset()
            if not self.recorder:
                config = dict(self.config, driver=self.driver)
                self.recorder = SessionRecorder(self.session_directory, config, self.provenance)
                self.last_session = self.recorder.id
            self.mode = 'running'
        elif kind == 'pause':
            if self.mode == 'running':
                self.mode = 'paused'
            self.pending.clear()
            self.held_keys.clear()
        elif kind == 'reset':
            self.reset()
        elif kind == 'erase_memory':
            if not self.brain or not self.plasticity.enabled:
                raise ValueError('No plastic memory is loaded')
            self.plasticity.erase(self.brain)
            self.reset()
        elif kind == 'reset_training':
            if self.trainer and self.trainer.template.get('action_mode') == 'held_sigmoid':
                self.trainer.template.pop('action_mode', None)
                self.trainer.template.pop('visual_policy', None)
                self.trainer.threshold = .45
                self.trainer.release = .2
            self.live_training_enabled = True
            self.init_trainer_from_scratch()
            self.reset()
        elif kind == 'save_checkpoint':
            saved_name = None
            if self.trainer is not None:
                live_dir = ROOT / 'outputs/training/live-interactive'
                live_dir.mkdir(parents=True, exist_ok=True)
                ckpt_name = f"episode-{self.generation:04d}.npz"
                ckpt_path = live_dir / ckpt_name
                self.trainer.save(ckpt_path)
                prog_file = live_dir / 'progress.json'
                ep_list = []
                for i, h in enumerate(self.training_history):
                    ep_list.append({
                        'index': h.get('generation', i + 1),
                        'name': f"episode-{h.get('generation', i + 1):04d}",
                        'score': h.get('score', 0),
                        'hits': h.get('hits', 0),
                        'misses': h.get('misses', 0),
                        'wrong': h.get('wrong', 0),
                        'accuracy': h.get('accuracy', 0.0),
                    })
                if not ep_list:
                    st = self.environment.get_state()
                    judged = st['hits'] + st['misses']
                    acc = round(100.0 * st['hits'] / judged, 1) if judged > 0 else 0.0
                    ep_list.append({
                        'index': self.generation,
                        'name': f"episode-{self.generation:04d}",
                        'score': st['score'],
                        'hits': st['hits'],
                        'misses': st['misses'],
                        'wrong': st['wrong'],
                        'accuracy': acc,
                    })
                prog_file.write_text(json.dumps({
                    'kind': 'live_interactive_training',
                    'episodes': ep_list
                }, indent=2))
                saved_name = f"outputs/training/live-interactive/{ckpt_name}"
            self.last_saved_checkpoint = saved_name
        elif kind == 'load_checkpoint':
            from training.checkpoints import resolve_checkpoint
            from training.rl import ReinforcementReadoutTrainer
            self._drain_neural_job()
            ckpt_path = resolve_checkpoint(command.get('path', ''))
            loaded = ReinforcementReadoutTrainer.load(ckpt_path)
            if self.brain is None:
                self.load_brain()
            if not calibration_graph_matches(GRAPH, loaded.template['graph_sha256'], self.provenance['graph_sha256']):
                raise ValueError('Checkpoint graph mismatch')
            if loaded.template.get('sensor') != self.decoder.record.get('sensor'):
                raise ValueError('Checkpoint sensor mismatch')
            if not (self.brain.state.ids[loaded.template['indices']].astype(str) == loaded.template['source_ids']).all():
                raise ValueError('Checkpoint neuron IDs mismatch')
            registry.create('decoder', 'calibrated', model=loaded.model())
            if loaded.template.get('song_sha256'):
                song_file = ROOT/'config/songs/la_consentida.json'
                if hashlib.sha256(song_file.read_bytes()).hexdigest() != loaded.template['song_sha256']:
                    raise ValueError('Checkpoint song mismatch')
                self.environment.load_song(json.loads(song_file.read_text()))
            self.trainer = loaded
            self.generation = len(loaded.episodes)
            self.training_history = []
            self.live_training_enabled = False
            self.driver = 'neural'
            self.provenance['decoder_sha256'] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
            self.reset()
        elif kind == 'select_song':
            song_id = command.get('song_id')
            song_file = ROOT / f'config/songs/{song_id}.json'
            if song_file.exists():
                song_data = json.loads(song_file.read_text())
                if hasattr(self.environment, 'load_song'):
                    self.environment.load_song(song_data)
                    self.reset()
        elif kind == 'keys' and self.driver == 'manual' and self.mode == 'running':
            self.pending.update(command['lanes'])
        elif kind == 'held_keys' and self.driver == 'manual':
            keys = set(command['lanes']) if self.mode == 'running' else set()
            self.pending.update(keys - self.held_keys)
            self.held_keys = keys
        elif kind == 'speed':
            self.clock.speed = float(command['value'])
        self.publish()

    def tick(self):
        import numpy as np
        import math
        activity = None
        started = time.perf_counter()

        t = getattr(self.environment, 'time', 0.0)
        bpm = getattr(self.environment, 'bpm', 108)
        beat_period = 60.0 / max(1, bpm)
        phase = (t % beat_period) / beat_period
        pulse = math.exp(-phase * 9.0) + 0.5 * math.exp(-((phase - 0.5) % 1.0) * 9.0)
        self.acoustic_energy = float(np.clip(pulse, 0.0, 1.0))

        if (self.driver == 'neural' and self.realtime_neural and
                self.trainer is not None and not self.live_training_enabled and
                not self.plasticity.enabled):
            self._tick_realtime_neural(started)
            return

        if self.driver == 'neural':
            stimulus = self.encoder.encode(self.environment.get_observation(), self.clock.neural_step_ms)
            aud_local = self.geometry.get('auditory_indices', [])
            if len(aud_local) > 0 and self.brain:
                aud_indices = np.array(self.geometry['indices'], dtype=np.int32)[aud_local]
                aud_currents = np.full(len(aud_indices), 18.0 * self.acoustic_energy + 2.0, dtype=np.float32)
                stimulus = NeuralStimulus(
                    indices=np.concatenate([stimulus.indices, aud_indices]),
                    currents=np.concatenate([stimulus.currents, aud_currents])
                )
            activity = self.brain.advance(stimulus, self.clock.neural_step_ms)
            if self.trainer is not None and self.live_training_enabled:
                norm_x = self.trainer.extract_features(activity)
                current_time_ms = self.environment.time * 1000.0
                abstract, rec = self.trainer.step(norm_x, current_time_ms, explore=True)
                self.current_trajectory.append(rec)
            elif self.trainer is not None:
                norm_x = self.trainer.extract_features(activity)
                abstract, _ = self.trainer.step(norm_x, self.environment.time * 1000., explore=False)
            else:
                abstract = self.decoder.decode(activity)
        else:
            abstract = Action(tuple(float(i in self.pending or i in self.held_keys) for i in range(4)))
            self.pending.clear()

        prev_hits = [l['hits'] for l in self.environment.lanes]
        prev_wrong = [l['wrong'] for l in self.environment.lanes]
        prev_misses = [l['misses'] for l in self.environment.lanes]

        self.environment.step(self.body.apply(abstract), self.clock.game_step)

        if self.driver == 'neural' and self.trainer is not None and self.live_training_enabled:
            step_r = np.zeros(4, dtype=np.float64)
            for lane in range(4):
                dh = self.environment.lanes[lane]['hits'] - prev_hits[lane]
                dw = self.environment.lanes[lane]['wrong'] - prev_wrong[lane]
                dm = self.environment.lanes[lane]['misses'] - prev_misses[lane]
                if dh > 0:
                    ev = self.environment.lanes[lane].get('last_event')
                    step_r[lane] += (1.0 if ev and ev.get('kind') == 'perfect' else 0.6) * dh
                if dw > 0:
                    step_r[lane] -= 0.1 * dw
                if dm > 0:
                    step_r[lane] -= 1.0 * dm
            self.current_rewards_by_lane.append(step_r)

        self.clock.advance(neural=activity is not None)
        if activity is not None and self.plasticity.enabled:
            self.plasticity.apply(self.brain, self.environment.get_reward())
        telemetry = self.metrics.record(activity, time.perf_counter() - started, self.environment.get_observation().timestamp)

        if self.environment.is_done():
            if self.driver == 'neural' and self.live_training_enabled and self.trainer is not None:
                # 1. Update policy
                if len(self.current_trajectory) > 0:
                    self.trainer.update_policy(self.current_trajectory, np.asarray(self.current_rewards_by_lane))
                
                # 2. Record generation
                st = self.environment.get_state()
                judged = st['hits'] + st['misses']
                acc = round(100.0 * st['hits'] / judged, 1) if judged > 0 else 0.0
                gen_record = {
                    'generation': self.generation,
                    'score': st['score'],
                    'accuracy': acc,
                    'hits': st['hits'],
                    'misses': st['misses'],
                    'wrong': st['wrong'],
                    'best_combo': st['best_combo'],
                    'seed': getattr(self.environment, 'seed', 0)
                }
                self.training_history.append(gen_record)
                
                # 3. Advance generation and seed
                self.generation += 1
                curr_seed = getattr(self.environment, 'seed', 0) or 0
                next_seed = curr_seed + 1
                
                # 4. Reset episode environment and runtime
                self.environment.reset(next_seed)
                self.clock.reset()
                self.metrics = Metrics()
                self.pending.clear()
                if self.brain:
                    self.brain.reset()
                    self.encoder.reset()
                self.trainer.reset_runtime()
                self.current_trajectory = []
                self.current_rewards_by_lane = []
                self.mode = 'running'  # Continuous learning without pausing!
            else:
                self.mode = 'finished'

        self.publish(activity, telemetry)
        self._record_frame()

    def publish(self, activity=None, telemetry=None):
        # Rendering is independent of neural computation and shows the exact sensory frame.
        image = Image.fromarray(self.environment.get_observation().rgb)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        prior = self.snapshot
        self.revision += 1
        rates = [0]*4
        if self.driver == 'neural':
            if self.trainer is not None and hasattr(self.trainer, 'last_rates'):
                rates = self.trainer.last_rates.tolist()
            elif self.decoder:
                rates = self.decoder.rates.tolist()

        self.snapshot = {'revision':self.revision, 'mode':self.mode, 'driver':self.driver,
            'game':self.environment.get_state(), 'error':self.error, 'speed':self.clock.speed,
            'frame':base64.b64encode(buffer.getvalue()).decode(),
            'live_training':{
                'enabled':self.live_training_enabled,
                'generation':self.generation,
                'history':self.training_history,
                'is_training':self.driver == 'neural' and self.trainer is not None and self.live_training_enabled,
                'method':self.trainer.template.get('training_method', 'RL interactivo') if self.trainer else None
            },
            'neural':{'backend':self.brain.backend if self.brain else 'not-loaded',
                      'neurons':self.brain.n if self.brain else self.geometry['total_neurons'],
                      'edges':len(self.brain.state.post) if self.brain else 25582938,
                      'time_ms':self.clock.neural_ms, 'loaded':self.brain is not None,
                      'fallback_reason':self.brain.fallback_reason if self.brain else None,
                      'readout_kind':self.config['decoder']['type'],
                      'readout_labels':['Lámina · '+key for key in 'DFJK'] if self.config['decoder']['type']=='calibrated' else ['DNa02 · L','DNpe017 · L','DNpe017 · R','DNa02 · R'],
                      'readout_rates':rates,
                      'acoustic_energy':round(getattr(self, 'acoustic_energy', 0.0), 3),
                      'sample_counts':activity.counts[self.geometry['indices']].tolist() if activity else prior.get('neural',{}).get('sample_counts', [0]*len(self.geometry['indices'])) if self.clock.ticks else [0]*len(self.geometry['indices'])},
            'telemetry':telemetry or (self.metrics.history[-1] if self.metrics.history else {'spikes':0,'total_spikes':0,'active_neurons':0,'step_wall_ms':0,'game_seconds':0,'neural_ms':0}),
            'session':self.last_session,
            'learning':self.plasticity.status(self.brain) if self.brain else self.plasticity.status(),
            'memory_checkpoint':self.last_memory}

    def loop(self):
        while not self.shutdown_event.is_set():
            started = time.perf_counter()
            try:
                for _ in range(128):
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    self.control(command)
                if self.mode == 'running':
                    self.tick()
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.error = f'{type(error).__name__}: {error}'
                self.mode = 'error'
                if self.recorder:
                    self.recorder.close()
                    self.recorder = None
                self.publish()
            period = self.clock.game_step / self.clock.speed if self.mode == 'running' else .02
            self.shutdown_event.wait(max(.001, period - (time.perf_counter() - started)))

    def close(self):
        self.shutdown_event.set()
        self.thread.join(timeout=10)
        self._drain_neural_job()
        self.neural_executor.shutdown(wait=True, cancel_futures=True)
        if self.recorder:
            self.recorder.close()
