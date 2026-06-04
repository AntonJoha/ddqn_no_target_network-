import argparse
from dataclasses import dataclass

import torch


def state_dict_equal(sd1, sd2):
    if sd1.keys() != sd2.keys():
        return False
    return all(torch.equal(sd1[k], sd2[k]) for k in sd1)


def state_dict_not_equal(sd1, sd2):
    return not state_dict_equal(sd1, sd2)

@dataclass
class DDQNConfig:
    env_id: str = "CartPole-v1"
    episodes: int = 5000
    max_steps: int = 1000
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 256
    replay_size: int = 50_000
    min_replay_size: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997
    target_update_freq: int = 200
    seed: int = 423354
    hidden_dim: int = 128
    eval_episodes: int = 10
    eval_seed_offset: int = 100_000
    render: bool = False
    target_network_countdown: int = 2500  # Steps before switching to target network
    lr_factor: float = 2
    lr_lower: float = 0.0001
    save_after: int=50
    save_rate: int=20
    save_before: int =500
    loss_threshold: int = 10
    current_episode: int = 1
    path: str | None = None
    replay_buffer_path: str | None = None




def parse_args() -> DDQNConfig:
    parser = argparse.ArgumentParser(description="Standard DDQN for LunarLander")
    parser.add_argument("--env-id", type=str, default="LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--min-replay-size", type=int, default=1_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed-offset", type=int, default=100_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--target-network-countdown", type=int, default=25)
    parser.add_argument("--replay-buffer-path", type=str, default=None)
    args = parser.parse_args()
    return DDQNConfig(
        env_id=args.env_id,
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        min_replay_size=args.min_replay_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        eval_episodes=args.eval_episodes,
        eval_seed_offset=args.eval_seed_offset,
        render=args.render,
        replay_buffer_path=args.replay_buffer_path,
    )

