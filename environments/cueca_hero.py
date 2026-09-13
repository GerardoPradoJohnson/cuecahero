"""Deterministic four-lane rhythm environment; no neural dependency."""
import math
import random
from PIL import Image, ImageDraw
import numpy as np
from core.contracts import Action, Observation, RewardSignal

COLORS = ['#c292ff', '#75d0b1', '#f0bd70', '#e88e9c']

class CuecaHeroEnvironment:
    def __init__(self, bpm=108, bars=12, lookahead=2.4, hit_window=.15):
        if not 40 <= bpm <= 200 or not 1 <= bars <= 64 or not .05 <= hit_window <= .3:
            raise ValueError('Invalid rhythm parameters')
        self.bpm, self.bars, self.lookahead, self.hit_window = bpm, bars, lookahead, hit_window
        self.custom_song = None
        self.reset()

    @property
    def song(self):
        return {
            'id': self.custom_song.get('id', 'primer_panuelo') if self.custom_song else 'primer_panuelo',
            'title': self.custom_song.get('title', 'Primer pañuelo') if self.custom_song else 'Primer pañuelo',
            'subtitle': self.custom_song.get('subtitle', 'PISTA ORIGINAL DE PRUEBA') if self.custom_song else 'PISTA ORIGINAL DE PRUEBA',
            'artist': self.custom_song.get('artist', 'Laboratorio FlyLab') if self.custom_song else 'Laboratorio FlyLab',
            'audio_url': self.custom_song.get('audio_url', '/audio/primer_panuelo.wav') if self.custom_song else '/audio/primer_panuelo.wav',
        }

    def load_song(self, song_data):
        """Load a charted song from catalog or file."""
        import copy
        self.custom_song = copy.deepcopy(song_data)
        self.bpm = int(song_data.get('bpm', self.bpm))
        self.reset(self.seed if hasattr(self, 'seed') else 0)
        return self.get_state()

    def reset(self, seed=0):
        self.seed = seed
        if self.custom_song:
            import copy
            self.bpm = int(self.custom_song.get('bpm', self.bpm))
            self.notes = copy.deepcopy(self.custom_song['notes'])
            for n in self.notes:
                n['judgement'] = None
            self.duration = float(self.custom_song.get('duration', self.notes[-1]['at'] + 1.2))
        else:
            rng = random.Random(seed)
            pulse = 60 / self.bpm / 3  # Two dotted-quarter beats per 6/8 bar.
            self.notes = []
            for bar in range(self.bars):
                for p in (0, 2, 3, 5):
                    self.notes.append({'id':len(self.notes), 'lane':(bar + p // 2 + rng.randrange(2)) % 4,
                                       'at':round(2.4 + (bar * 6 + p) * pulse, 9), 'judgement':None})
            self.duration = self.notes[-1]['at'] + 1.2
        self.time = 0.
        self.score = self.combo = self.best_combo = self.hits = self.misses = self.wrong = 0
        self.events = []
        self.lanes = [dict(hits=0, misses=0, wrong=0, last_event=None) for _ in range(4)]
        self.active_holds = {}
        self.last_action = [0] * 4
        self.reward = RewardSignal(0., 'cueca_hero', 0.)
        return self.get_state()

    def step(self, action: Action, dt: float):
        if not math.isfinite(dt) or not 0 < dt <= .2:
            raise ValueError('Environment dt must be in (0, 0.2] seconds')
        if len(action.values) != 4 or not all(math.isfinite(x) for x in action.values):
            raise ValueError('Four finite controls required')
        if self.is_done():
            self.reward = RewardSignal(0., 'cueca_hero', self.time)
            return self.get_state()
        self.time = min(self.time + dt, self.duration)
        self.events = []
        reward = 0.
        self.last_action = [int(x > .5) for x in action.values]

        # 1. Process active sustain holds
        for lane in list(self.active_holds.keys()):
            hold = self.active_holds[lane]
            if self.last_action[lane]:
                # Continues holding
                multiplier = min(4, 1 + (self.combo - 1) // 8) if self.combo > 0 else 1
                hold_pts = 4 * multiplier
                self.score += hold_pts
                reward += 0.04
                hold['ticks'] += 1
                if self.time >= hold['end_at']:
                    self.events.append({'kind':'good', 'lane':lane, 'note':hold['note_id'], 'hold_complete':True})
                    del self.active_holds[lane]
            else:
                # Released early
                del self.active_holds[lane]

        # 2. Process new keypresses
        for lane, pressed in enumerate(self.last_action):
            if not pressed or lane in self.active_holds:
                continue
            candidates = [n for n in self.notes if n['lane'] == lane and n['judgement'] is None and abs(n['at'] - self.time) <= self.hit_window + 1e-9]
            if candidates:
                note = min(candidates, key=lambda n:abs(n['at'] - self.time))
                error = self.time - note['at']
                perfect = abs(error) <= .065 + 1e-9
                note['judgement'] = 'perfect' if perfect else 'good'
                self.combo += 1
                self.best_combo = max(self.best_combo, self.combo)
                self.score += (100 if perfect else 60) * min(4, 1 + (self.combo - 1) // 8)
                self.hits += 1
                reward += 1. if perfect else .6
                self.events.append({'kind':note['judgement'], 'lane':lane, 'note':note['id'], 'error_ms':round(error * 1000, 2)})
                # Register hold if note has sustain
                sustain = float(note.get('sustain', 0.0))
                if sustain > 0.12:
                    self.active_holds[lane] = {
                        'note_id': note['id'],
                        'end_at': note['at'] + sustain,
                        'ticks': 0
                    }
            else:
                self.combo = 0
                self.wrong += 1
                reward -= .1
                self.events.append({'kind':'empty', 'lane':lane})
        for note in self.notes:
            if note['judgement'] is None and self.time > note['at'] + self.hit_window + 1e-9:
                note['judgement'] = 'miss'
                self.misses += 1
                self.combo = 0
                reward -= 1.
                self.events.append({'kind':'miss', 'lane':note['lane'], 'note':note['id']})
        for event in self.events:
            lane = self.lanes[event['lane']]
            lane['hits' if event['kind'] in ('perfect', 'good') else 'misses' if event['kind'] == 'miss' else 'wrong'] += 1
            lane['last_event'] = dict(event, time=self.time)
        self.reward = RewardSignal(reward, 'cueca_hero', self.time)
        return self.get_state()

    def get_observation(self):
        # This exact image is also displayed in the browser. Only visible notes.
        w, h = 320, 400
        image = Image.new('RGB', (w, h), '#151722')
        draw = ImageDraw.Draw(image)
        target = 332
        for lane in range(4):
            x = lane * 80
            draw.rectangle((x + 1, 0, x + 79, h), fill='#1b1d2a' if lane % 2 else '#181a26')
            draw.line((x, 0, x, h), fill='#313140')
            is_holding = lane in self.active_holds
            draw.rounded_rectangle((x + 10, target - 12, x + 70, target + 12), radius=6,
                                   fill=COLORS[lane] if (self.last_action[lane] or is_holding) else '#272837', outline=COLORS[lane], width=2)
            key_name = ['D', 'F', 'J', 'K'][lane]
            draw.text((x + 36, target - 6), key_name, fill='#ffffff' if (self.last_action[lane] or is_holding) else '#8a8398')
        pulse = 60 / self.bpm / 3
        first = math.floor((self.time - 2.4) / pulse)
        for p in range(first, first + math.ceil(self.lookahead / pulse) + 2):
            at = 2.4 + p * pulse
            y = target - (at - self.time) / self.lookahead * target
            if 0 <= y <= h:
                draw.line((0, int(y), w, int(y)), fill='#343140' if p % 3 == 0 else '#202230')

        # 1. Draw sustain ribbons first (so heads render on top)
        for note in self.notes:
            sustain = float(note.get('sustain', 0.0))
            if sustain > 0.1:
                delta = note['at'] - self.time
                end_delta = (note['at'] + sustain) - self.time
                if note['judgement'] != 'miss' and delta <= self.lookahead and end_delta >= -self.hit_window:
                    x = note['lane'] * 80
                    y_head = target - delta / self.lookahead * target
                    y_tail = target - end_delta / self.lookahead * target
                    y_bottom = max(0, min(h, int(y_head)))
                    y_top = max(0, min(h, int(y_tail)))
                    if y_bottom > y_top:
                        col = COLORS[note['lane']]
                        draw.rectangle((x + 31, y_top, x + 49, y_bottom), fill=col)
                        draw.rectangle((x + 36, y_top, x + 44, y_bottom), fill='#ffffff')
                        # Cap at tail end
                        if 0 <= y_tail <= h:
                            draw.rounded_rectangle((x + 28, int(y_tail) - 3, x + 52, int(y_tail) + 3), radius=3, fill=col)

        # 2. Draw note heads
        for note in self.notes:
            delta = note['at'] - self.time
            if note['judgement'] is None and -self.hit_window <= delta <= self.lookahead:
                x = note['lane'] * 80
                y = target - delta / self.lookahead * target
                draw.rounded_rectangle((x + 12, int(y) - 7, x + 68, int(y) + 7), radius=4, fill=COLORS[note['lane']])
                draw.line((x + 19, int(y) - 3, x + 61, int(y) - 3), fill='#f6edf8', width=2)
        return Observation(np.asarray(image), self.time)

    def get_reward(self):
        return self.reward

    def is_done(self):
        return self.time >= self.duration

    def get_state(self):
        judged = self.hits + self.misses
        song_info = {
            'id': self.custom_song.get('id', 'primer_panuelo') if self.custom_song else 'primer_panuelo',
            'title': self.custom_song.get('title', 'Primer pañuelo') if self.custom_song else 'Primer pañuelo',
            'subtitle': self.custom_song.get('subtitle', 'PISTA ORIGINAL DE PRUEBA') if self.custom_song else 'PISTA ORIGINAL DE PRUEBA',
            'artist': self.custom_song.get('artist', 'Laboratorio FlyLab') if self.custom_song else 'Laboratorio FlyLab',
            'audio_url': self.custom_song.get('audio_url', '/audio/primer_panuelo.wav') if self.custom_song else '/audio/primer_panuelo.wav',
        }
        return {'environment':'cueca_hero', 'time':round(self.time, 6), 'duration':self.duration,
                'score':self.score, 'combo':self.combo, 'best_combo':self.best_combo,
                'hits':self.hits, 'misses':self.misses, 'wrong':self.wrong,
                'accuracy':round(100 * self.hits / judged, 1) if judged else None,
                'total_notes':len(self.notes), 'events':self.events, 'action':self.last_action,
                'active_holds':[lane in self.active_holds for lane in range(4)],
                'lanes':[dict(lane, last_event=dict(lane['last_event']) if lane['last_event'] else None) for lane in self.lanes],
                'done':self.is_done(), 'bpm':self.bpm, 'seed':self.seed, 'reward':self.reward.magnitude,
                'song':song_info}
