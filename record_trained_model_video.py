import argparse
import dataclasses
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from ddqn_lunar_lander import QNetwork, device, set_seed
from util import DDQNConfig


def load_config(config_path: Path | None, model_path: Path, cli_overrides: dict[str, object]) -> DDQNConfig:
    candidate = config_path or model_path.with_suffix(".json")
    raw_config: dict[str, object] = {}
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as file:
            raw_config = json.load(file)

    valid_fields = {field.name for field in dataclasses.fields(DDQNConfig)}
    config_data = {key: value for key, value in raw_config.items() if key in valid_fields}
    config_data.update({key: value for key, value in cli_overrides.items() if value is not None})
    config = DDQNConfig(**config_data)
    config.path = str(model_path)
    return config


def load_policy(model_path: Path, config: DDQNConfig, state_dim: int, action_dim: int) -> QNetwork:
    policy = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a trained DDQN rollout as video")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a trained .pth checkpoint")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional path to the matching training config JSON",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("videos"), help="Directory for video files")
    parser.add_argument("--output-name", type=str, default=None, help="Video file prefix")
    parser.add_argument("--episodes", type=int, default=1, help="How many episodes to record")
    parser.add_argument("--max-steps", type=int, default=None, help="Override the max steps per episode")
    parser.add_argument("--seed", type=int, default=None, help="Override the rollout seed")
    parser.add_argument("--env-id", type=str, default=None, help="Override the environment id")
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(
        args.config_path,
        args.model_path,
        {
            "max_steps": args.max_steps,
            "seed": args.seed,
            "env_id": args.env_id,
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    env = gym.make(config.env_id, render_mode="rgb_array")
    rollout_seed = int(np.uint32(config.seed if config.seed is not None else 0))
    if not isinstance(env.observation_space, gym.spaces.Box) or not isinstance(
        env.action_space, gym.spaces.Discrete
    ):
        raise TypeError("Only Box observation spaces and Discrete action spaces are supported.")

    env.unwrapped.metadata["render_fps"] = args.fps
    env.action_space.seed(rollout_seed)
    video_name_prefix = args.output_name or args.model_path.stem
    env = RecordVideo(
        env,
        video_folder=str(args.output_dir),
        name_prefix=video_name_prefix,
        episode_trigger=lambda _episode_id: True,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = load_policy(args.model_path, config, state_dim, action_dim)

    rewards: list[float] = []
    try:
        for episode in range(args.episodes):
            episode_seed = int(np.uint32(rollout_seed + episode))
            state, _ = env.reset(seed=episode_seed)
            episode_reward = 0.0
            for _ in range(config.max_steps):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.inference_mode():
                    action = int(policy(state_t).argmax(dim=1).item())
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: reward={episode_reward:.2f}")
    finally:
        env.close()

    video_files = sorted(
        path
        for path in args.output_dir.glob(f"{video_name_prefix}*")
        if path.suffix.lower() in {".mp4", ".webm", ".avi"}
    )
    if video_files:
        print(f"Video saved to: {video_files[-1]}")
    print(f"Mean reward: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    main()
