import argparse
import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class DDQNConfig:
    env_id: str = "LunarLander-v3"
    episodes: int = 500
    max_steps: int = 1000
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 100_000
    min_replay_size: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_freq: int = 200
    seed: int = 42
    hidden_dim: int = 128
    eval_episodes: int = 5
    eval_seed_offset: int = 1_000_000
    render: bool = False


class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: DDQNConfig, device: torch.device):
        self.device = device
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.update_steps = 0

        self.policy_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state: np.ndarray, epsilon: float, action_dim: int) -> int:
        if random.random() < epsilon:
            return random.randrange(action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, replay_buffer: ReplayBuffer):
        if len(replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: DDQNConfig):
    set_seed(config.seed)
    render_mode = "human" if config.render else None
    env = gym.make(config.env_id, render_mode=render_mode)
    env.action_space.seed(config.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DDQNAgent(state_dim, action_dim, config, device)
    replay_buffer = ReplayBuffer(config.replay_size)
    epsilon = config.epsilon_start
    episode_rewards = []

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)
        episode_reward = 0.0

        for _ in range(config.max_steps):
            action = agent.act(state, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= config.min_replay_size:
                agent.train_step(replay_buffer)

            if done:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        episode_rewards.append(episode_reward)
        avg_last_10 = np.mean(episode_rewards[-10:])
        print(
            f"Episode {episode:4d}/{config.episodes} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Avg(10): {avg_last_10:8.2f} | "
            f"Epsilon: {epsilon:6.3f}"
        )

    env.close()
    return agent, device


def evaluate(agent: DDQNAgent, config: DDQNConfig, device: torch.device):
    env = gym.make(config.env_id, render_mode="human" if config.render else None)
    rewards = []
    for episode in range(1, config.eval_episodes + 1):
        state, _ = env.reset(seed=config.seed + config.eval_seed_offset + episode)
        total_reward = 0.0
        for _ in range(config.max_steps):
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(agent.policy_net(state_t).argmax(dim=1).item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        print(f"[Eval] Episode {episode}: {total_reward:.2f}")
    env.close()
    print(f"[Eval] Mean reward over {config.eval_episodes} episodes: {np.mean(rewards):.2f}")


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
    parser.add_argument("--eval-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--render", action="store_true")
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
    )


if __name__ == "__main__":
    cfg = parse_args()
    trained_agent, device = train(cfg)
    evaluate(trained_agent, cfg, device)
