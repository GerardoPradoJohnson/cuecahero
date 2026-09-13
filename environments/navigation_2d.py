"""Continuous 2D arena navigation and obstacle avoidance environment; no neural dependency."""
import math
import random
from typing import Any, Mapping
from PIL import Image, ImageDraw
import numpy as np
from core.contracts import Action, Observation, MultimodalObservation, RewardSignal

ARENA_WIDTH = 320
ARENA_HEIGHT = 400
ARENA_MARGIN = 18

COLORS = {
    'background': '#13151f',
    'arena_bg': '#191b28',
    'grid': '#212435',
    'wall': '#3a3e5c',
    'wall_glow': '#272a40',
    'agent_body': '#c292ff',
    'agent_wing': '#e0c7ff',
    'agent_heading': '#ffffff',
    'agent_fov': '#c292ff22',
    'target': '#75d0b1',
    'target_glow': '#75d0b144',
    'obstacle': '#e88e9c',
    'obstacle_inner': '#f5b0bc',
    'hud_text': '#a5a7ba',
    'hud_highlight': '#f0bd70',
}


class Navigation2DEnvironment:
    """A 2D spatial navigation arena with continuous kinematics, food targets,

    and obstacles. Fully decoupled from neural computation.
    """

    def __init__(
        self,
        duration: float = 30.0,
        max_speed: float = 90.0,
        accel: float = 160.0,
        brake: float = 200.0,
        turn_rate: float = 4.0,
        friction: float = 1.2,
        num_obstacles: int = 5,
        target_radius: float = 11.0,
        obstacle_radius: float = 14.0,
        agent_radius: float = 8.0,
    ):
        if not 0.5 <= duration <= 600.0:
            raise ValueError('Duration must be between 0.5 and 600 seconds')
        if not 10.0 <= max_speed <= 300.0:
            raise ValueError('Invalid speed parameters')
        self.duration = duration
        self.max_speed = max_speed
        self.accel = accel
        self.brake = brake
        self.turn_rate = turn_rate
        self.friction = friction
        self.num_obstacles = num_obstacles
        self.target_radius = target_radius
        self.obstacle_radius = obstacle_radius
        self.agent_radius = agent_radius
        self.reset()

    def reset(self, seed: int = 0) -> Mapping[str, Any]:
        self.seed = seed
        self.rng = random.Random(seed)
        self.time = 0.0
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.hits = 0      # Target pickups
        self.misses = 0    # Obstacle collisions
        self.wrong = 0     # Wall bumps

        # Agent initial state: center of arena pointing upwards (-pi/2)
        self.x = ARENA_WIDTH / 2.0
        self.y = ARENA_HEIGHT * 0.7
        self.angle = -math.pi / 2.0
        self.speed = 0.0
        self.trail: list[tuple[float, float]] = []

        # Obstacles generation
        self.obstacles: list[dict[str, float]] = []
        for i in range(self.num_obstacles):
            ox = self.rng.uniform(ARENA_MARGIN + 30, ARENA_WIDTH - ARENA_MARGIN - 30)
            oy = self.rng.uniform(ARENA_MARGIN + 40, ARENA_HEIGHT * 0.55)
            self.obstacles.append({
                'id': i,
                'x': ox,
                'y': oy,
                'r': self.rng.uniform(self.obstacle_radius * 0.8, self.obstacle_radius * 1.3),
                'vx': self.rng.uniform(-15.0, 15.0),
                'vy': self.rng.uniform(-8.0, 8.0),
            })

        # Target initialization
        self.target = {'x': 0.0, 'y': 0.0, 'r': self.target_radius}
        self._spawn_target()

        self.last_distance = self._distance_to_target()
        self.events: list[dict[str, Any]] = []
        self.last_action = [0, 0, 0, 0]
        self.lanes = [
            dict(hits=0, misses=0, wrong=0, last_event=None) for _ in range(4)
        ]
        self.reward = RewardSignal(0.0, 'navigation_2d', 0.0)
        return self.get_state()

    def _spawn_target(self) -> None:
        """Place a target away from obstacles and arena borders."""
        for _ in range(50):
            tx = self.rng.uniform(ARENA_MARGIN + 25, ARENA_WIDTH - ARENA_MARGIN - 25)
            ty = self.rng.uniform(ARENA_MARGIN + 25, ARENA_HEIGHT - ARENA_MARGIN - 25)
            # Avoid placing directly on agent or obstacles
            if math.hypot(tx - self.x, ty - self.y) < 50.0:
                continue
            if any(math.hypot(tx - ob['x'], ty - ob['y']) < ob['r'] + 20 for ob in self.obstacles):
                continue
            self.target['x'] = tx
            self.target['y'] = ty
            return
        # Fallback placement
        self.target['x'] = ARENA_WIDTH / 2.0
        self.target['y'] = ARENA_MARGIN + 45.0

    def _distance_to_target(self) -> float:
        return math.hypot(self.target['x'] - self.x, self.target['y'] - self.y)

    def step(self, action: Action, dt: float) -> Mapping[str, Any]:
        if not math.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError('Environment dt must be in (0, 0.2] seconds')
        if len(action.values) != 4 or not all(math.isfinite(x) for x in action.values):
            raise ValueError('Four finite controls required: [Left, Forward, Right, Brake]')

        if self.is_done():
            self.reward = RewardSignal(0.0, 'navigation_2d', self.time)
            return self.get_state()

        self.time = min(self.time + dt, self.duration)
        self.events = []
        reward_magnitude = 0.0

        # Action interpretation:
        # [0]: Turn Left, [1]: Forward Thrust, [2]: Turn Right, [3]: Brake / Reverse
        # Supports both continuous analog values (e.g. 0.0 to 1.0) and discrete values
        a0, a1, a2, a3 = [float(x) for x in action.values]
        self.continuous_action = [a0, a1, a2, a3]
        self.last_action = [int(x > 0.5) for x in action.values]

        # Update heading using differential turning: (right - left)
        turn_delta = a2 - a0
        angular_vel = turn_delta * self.turn_rate
        self.angle += angular_vel * dt
        # Wrap angle to [-pi, pi]
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

        # Update speed with continuous thrust and braking
        if a1 > 0.0:
            self.speed = min(self.max_speed, self.speed + a1 * self.accel * dt)
        if a3 > 0.0:
            self.speed = max(0.0, self.speed - a3 * self.brake * dt)
        # Apply natural friction / fluid drag
        self.speed = max(0.0, self.speed * math.exp(-self.friction * dt))

        # Position update
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt

        # Update trail history
        if not self.trail or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) >= 6.0:
            self.trail.append((self.x, self.y))
            if len(self.trail) > 40:
                self.trail.pop(0)

        # Move dynamic obstacles
        min_x = ARENA_MARGIN + self.obstacle_radius
        max_x = ARENA_WIDTH - ARENA_MARGIN - self.obstacle_radius
        min_y = ARENA_MARGIN + self.obstacle_radius
        max_y = ARENA_HEIGHT - ARENA_MARGIN - self.obstacle_radius
        for ob in self.obstacles:
            ob['x'] += ob['vx'] * dt
            ob['y'] += ob['vy'] * dt
            if ob['x'] <= min_x:
                ob['x'] = min_x
                ob['vx'] = abs(ob['vx'])
            elif ob['x'] >= max_x:
                ob['x'] = max_x
                ob['vx'] = -abs(ob['vx'])
            if ob['y'] <= min_y:
                ob['y'] = min_y
                ob['vy'] = abs(ob['vy'])
            elif ob['y'] >= max_y:
                ob['y'] = max_y
                ob['vy'] = -abs(ob['vy'])

        # Wall collisions and boundary constraints
        wall_bump = False
        min_ax = ARENA_MARGIN + self.agent_radius
        max_ax = ARENA_WIDTH - ARENA_MARGIN - self.agent_radius
        min_ay = ARENA_MARGIN + self.agent_radius
        max_ay = ARENA_HEIGHT - ARENA_MARGIN - self.agent_radius

        if self.x < min_ax:
            self.x = min_ax
            wall_bump = True
        elif self.x > max_ax:
            self.x = max_ax
            wall_bump = True
        if self.y < min_ay:
            self.y = min_ay
            wall_bump = True
        elif self.y > max_ay:
            self.y = max_ay
            wall_bump = True

        if wall_bump:
            self.wrong += 1
            self.speed *= 0.3
            reward_magnitude -= 0.2
            self.events.append({'kind': 'empty', 'lane': 3 if a3 > 0.5 else 1})

        # Obstacle collisions
        collided = False
        for ob in self.obstacles:
            dist = math.hypot(self.x - ob['x'], self.y - ob['y'])
            min_dist = self.agent_radius + ob['r']
            if dist < min_dist:
                collided = True
                self.misses += 1
                self.combo = 0
                reward_magnitude -= 0.6
                # Bounce normal
                nx = (self.x - ob['x']) / max(0.1, dist)
                ny = (self.y - ob['y']) / max(0.1, dist)
                self.x = ob['x'] + nx * (min_dist + 1.0)
                self.y = ob['y'] + ny * (min_dist + 1.0)
                self.speed *= 0.2
                self.events.append({'kind': 'miss', 'lane': 0 if a0 > 0.5 else 2 if a2 > 0.5 else 1})
                break

        # Target pickup detection
        target_dist = self._distance_to_target()
        if target_dist <= self.agent_radius + self.target['r']:
            self.hits += 1
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
            bonus = min(4, 1 + (self.combo - 1) // 3)
            self.score += 100 * bonus
            reward_magnitude += 1.0
            self.events.append({'kind': 'perfect', 'lane': 1})
            self._spawn_target()
            self.last_distance = self._distance_to_target()
        else:
            # Subtle shaping reward for closing distance towards food target
            shaping = (self.last_distance - target_dist) * 0.015
            if not collided and not wall_bump:
                reward_magnitude += shaping
            self.last_distance = target_dist

        # Feed lane event feedback for UI visualizers
        for ev in self.events:
            lane = self.lanes[ev['lane']]
            if ev['kind'] in ('perfect', 'good'):
                lane['hits'] += 1
            elif ev['kind'] == 'miss':
                lane['misses'] += 1
            else:
                lane['wrong'] += 1
            lane['last_event'] = dict(ev, time=self.time)

        self.reward = RewardSignal(round(reward_magnitude, 4), 'navigation_2d', self.time)
        return self.get_state()

    def get_observation(self) -> Observation:
        """Render high-contrast visual raster of the arena, target, obstacles, and fly."""
        w, h = ARENA_WIDTH, ARENA_HEIGHT
        image = Image.new('RGB', (w, h), COLORS['background'])
        draw = ImageDraw.Draw(image)

        # Arena playing field
        margin = ARENA_MARGIN
        draw.rectangle((margin, margin, w - margin, h - margin), fill=COLORS['arena_bg'], outline=COLORS['wall'], width=2)

        # Grid lines (20px spacing)
        for gx in range(margin + 20, w - margin, 32):
            draw.line((gx, margin, gx, h - margin), fill=COLORS['grid'])
        for gy in range(margin + 20, h - margin, 32):
            draw.line((margin, gy, w - margin, gy), fill=COLORS['grid'])

        # Trail breadcrumbs
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int(255 * (i + 1) / max(1, len(self.trail)))
            r_crumb = 2 if i < len(self.trail) - 5 else 3
            draw.ellipse((tx - r_crumb, ty - r_crumb, tx + r_crumb, ty + r_crumb), fill='#7e62a8')

        # Obstacles
        for ob in self.obstacles:
            ox, oy, orad = ob['x'], ob['y'], ob['r']
            draw.ellipse((ox - orad, oy - orad, ox + orad, oy + orad), fill=COLORS['obstacle'], outline=COLORS['obstacle_inner'], width=2)
            # Center core
            draw.ellipse((ox - 3, oy - 3, ox + 3, oy + 3), fill='#ffffff')

        # Food / Recompense Target
        tx, ty, trad = self.target['x'], self.target['y'], self.target['r']
        # Outer pulsating halo
        halo = trad + 4 + math.sin(self.time * 6.0) * 2.0
        draw.ellipse((tx - halo, ty - halo, tx + halo, ty + halo), outline=COLORS['target'], width=1)
        draw.ellipse((tx - trad, ty - trad, tx + trad, ty + trad), fill=COLORS['target'], outline='#ffffff', width=2)
        draw.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), fill='#ffffff')

        # Agent (Fly) Avatar
        ax, ay = self.x, self.y
        rad = self.agent_radius

        # Field of View cone (60 degrees forward)
        fov_dist = 42.0
        fov_left = self.angle - math.radians(35)
        fov_right = self.angle + math.radians(35)
        lx = ax + math.cos(fov_left) * fov_dist
        ly = ay + math.sin(fov_left) * fov_dist
        rx = ax + math.cos(fov_right) * fov_dist
        ry = ay + math.sin(fov_right) * fov_dist
        draw.polygon([(ax, ay), (lx, ly), (rx, ry)], outline='#584570')

        # Fly Wings (drawn perpendicular to body)
        wing_ang1 = self.angle + math.pi / 2.0
        wing_ang2 = self.angle - math.pi / 2.0
        wlen = rad * 1.5
        w1x = ax + math.cos(wing_ang1) * wlen
        w1y = ay + math.sin(wing_ang1) * wlen
        w2x = ax + math.cos(wing_ang2) * wlen
        w2y = ay + math.sin(wing_ang2) * wlen
        draw.ellipse((w1x - 4, w1y - 3, w1x + 4, w1y + 3), fill=COLORS['agent_wing'])
        draw.ellipse((w2x - 4, w2y - 3, w2x + 4, w2y + 3), fill=COLORS['agent_wing'])

        # Fly Body
        draw.ellipse((ax - rad, ay - rad, ax + rad, ay + rad), fill=COLORS['agent_body'], outline='#ffffff', width=2)

        # Directional Heading Pointer
        hx = ax + math.cos(self.angle) * (rad + 6)
        hy = ay + math.sin(self.angle) * (rad + 6)
        draw.line((ax, ay, hx, hy), fill=COLORS['agent_heading'], width=2)

        # Minimalist in-canvas HUD (Top bar)
        draw.rectangle((0, 0, w, margin - 1), fill='#0d0e15')
        draw.line((0, margin - 1, w, margin - 1), fill='#2a2d40')
        hud = f'SCORE {self.score}  TARGETS {self.hits}  HITS {self.misses}  {self.time:.1f}s'
        draw.text((margin + 4, 3), hud, fill=COLORS['hud_text'])

        # Multimodal sensory signals
        proprio = {
            'speed': float(self.speed),
            'angular_velocity': float((self.last_action[2] - self.last_action[0]) * self.turn_rate),
            'collision': 1.0 if any(e['kind'] == 'miss' for e in self.events) else 0.0,
        }
        base_freq = 180.0 + self.speed * 1.5
        t_samples = np.linspace(0, 0.01, 16, endpoint=False, dtype=np.float32)
        audio_wave = np.sin(2 * math.pi * base_freq * t_samples) * 0.35
        if any(e['kind'] == 'miss' for e in self.events):
            audio_wave = audio_wave + np.full(16, 0.5, dtype=np.float32)

        return MultimodalObservation(np.asarray(image), self.time, audio=audio_wave, proprioception=proprio)

    def get_reward(self) -> RewardSignal:
        return self.reward

    def is_done(self) -> bool:
        return self.time >= self.duration

    def get_state(self) -> Mapping[str, Any]:
        judged = self.hits + self.misses
        accuracy = round(100.0 * self.hits / judged, 1) if judged > 0 else None
        return {
            'environment': 'navigation_2d',
            'time': round(self.time, 6),
            'duration': self.duration,
            'score': self.score,
            'combo': self.combo,
            'best_combo': self.best_combo,
            'hits': self.hits,
            'misses': self.misses,
            'wrong': self.wrong,
            'accuracy': accuracy,
            'action': self.last_action,
            'events': self.events,
            'lanes': [
                dict(lane, last_event=dict(lane['last_event']) if lane['last_event'] else None)
                for lane in self.lanes
            ],
            'done': self.is_done(),
            'seed': self.seed,
            'reward': self.reward.magnitude,
            'agent': {
                'x': round(self.x, 2),
                'y': round(self.y, 2),
                'angle_deg': round(math.degrees(self.angle), 1),
                'speed': round(self.speed, 2),
            },
            'target': {
                'x': round(self.target['x'], 2),
                'y': round(self.target['y'], 2),
                'distance': round(self._distance_to_target(), 2),
            },
            'obstacles_count': len(self.obstacles),
        }
