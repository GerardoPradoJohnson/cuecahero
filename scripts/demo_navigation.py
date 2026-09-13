"""Demonstration script: Run MaleCNSCore navigating in Navigation2DEnvironment."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.malecns import MaleCNSCore
from environments.navigation_2d import Navigation2DEnvironment
from sensors.visual import VisualEncoder
from decoders.directional import DirectionalDecoder
from embodiments.controller import GameController
from core.paths import DATA


def main():
    parser = argparse.ArgumentParser(description="Run 2D navigation demo with MaleCNSCore")
    parser.add_argument("--steps", type=int, default=60, help="Simulation steps to run")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--save-frame", type=Path, default=ROOT / "outputs/navigation_demo_frame.png")
    args = parser.parse_args()

    print("=" * 60)
    print("  MaleCNS 2D Navigation Demonstration")
    print("=" * 60)

    # 1. Initialize environment
    env = Navigation2DEnvironment(duration=30.0)
    env.reset(args.seed)
    body = GameController()

    # 2. Initialize MaleCNS
    print("Loading MaleCNS connectome...")
    brain = MaleCNSCore(backend="auto")
    print(f"MaleCNS backend: {brain.backend}, neurons: {brain.n}")

    # 3. Initialize visual sensor encoder
    encoder = VisualEncoder(retina=brain.state.retina, uv=brain.state.uv, lamina=brain.state.lamina)

    # 4. Initialize directional decoder from descending motor neurons
    manifest = json.loads((DATA / "outputs/doom/malecns_v1/manifest.json").read_text())["readouts"]
    specs = [
        {"type": "DNa02", "side": "L"},
        {"type": "DNpe017", "side": "L"},
        {"type": "DNpe017", "side": "R"},
        {"type": "DNa02", "side": "R"},
    ]
    groups = [[r["index"] for r in manifest if r["type"] == s["type"] and r["side"] == s["side"]] for s in specs]
    decoder = DirectionalDecoder(groups=groups, threshold_hz=0.8, cooldown_ms=60.0)

    print(f"Starting closed-loop simulation ({args.steps} steps @ 30 Hz)...")
    dt = 1.0 / 30.0
    neural_step_ms = 10.0

    total_spikes = 0
    actions_fired = [0, 0, 0, 0]

    for step in range(args.steps):
        obs = env.get_observation()
        stim = encoder.encode(obs, neural_step_ms)
        activity = brain.advance(stim, neural_step_ms)
        abstract_action = decoder.decode(activity)
        motor_action = body.apply(abstract_action)
        state = env.step(motor_action, dt)

        total_spikes += int(activity.counts.sum())
        for i, val in enumerate(motor_action.values):
            if val > 0.5:
                actions_fired[i] += 1

        if (step + 1) % 15 == 0 or step == args.steps - 1:
            agent = state["agent"]
            target = state["target"]
            reward = state["reward"]
            print(
                f"Step {step+1:02d}/{args.steps} | Pos: ({agent['x']:.1f}, {agent['y']:.1f}) | "
                f"Angle: {agent['angle_deg']:+5.1f}° | Speed: {agent['speed']:.1f} | "
                f"Target dist: {target['distance']:.1f} | Reward: {reward:+.3f} | Spikes: {activity.counts.sum()}"
            )

    print("-" * 60)
    final_state = env.get_state()
    print("Episode Summary:")
    print(f"  Total spikes generated : {total_spikes:,}")
    print(f"  Actions fired (L/F/R/B): {actions_fired}")
    print(f"  Targets captured       : {final_state['hits']}")
    print(f"  Obstacle collisions    : {final_state['misses']}")
    print(f"  Wall bumps             : {final_state['wrong']}")
    print(f"  Final score            : {final_state['score']}")

    # Save final visual observation frame
    args.save_frame.parent.mkdir(parents=True, exist_ok=True)
    obs = env.get_observation()
    from PIL import Image
    Image.fromarray(obs.rgb).save(args.save_frame)
    print(f"Saved observation snapshot to: {args.save_frame}")
    print("=" * 60)


if __name__ == "__main__":
    main()
