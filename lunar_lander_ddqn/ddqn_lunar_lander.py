import dataclasses
import json
import os
import pickle
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from sgberd import SGBerD_wrapper
from tgelu import TGeLU
from util import *

MAX_SEED_VALUE = 2**32 - 1
REPLAY_BUFFER_SUFFIX = ".replay.pkl"
TGELU_LEFT_THRESHOLD = -1
TGELU_RIGHT_THRESHOLD = 1
LOSS_MEAN_THRESHOLD = 0.5
NOISE_UPDATE_FREQUENCY = 100
NOISE_DECAY_FACTOR = 0.95


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            TGeLU(TGELU_LEFT_THRESHOLD, TGELU_RIGHT_THRESHOLD, device=device),
            nn.Linear(hidden_dim, hidden_dim),
            TGeLU(TGELU_LEFT_THRESHOLD, TGELU_RIGHT_THRESHOLD, device=device),
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
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str):
        try:
            with open(path, "wb") as file:
                pickle.dump(self, file)
        except pickle.PicklingError as error:
            raise ValueError(f"Failed to serialize replay buffer to {path}: {error}") from error
        except OSError as error:
            raise OSError(f"Failed to save replay buffer to {path}: {error}") from error

    @classmethod
    def load(cls, path: str):
        try:
            with open(path, "rb") as file:
                replay_buffer = pickle.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Replay buffer file not found at {path}: {error}") from error
        except pickle.UnpicklingError as error:
            raise ValueError(f"Failed to deserialize replay buffer from {path}: {error}") from error
        except OSError as error:
            raise OSError(f"Failed to load replay buffer from {path}: {error}") from error
        if not isinstance(replay_buffer, cls):
            raise TypeError(
                f"Loaded object is not a ReplayBuffer. Got {type(replay_buffer).__name__}."
            )
        return replay_buffer


class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: DDQNConfig, device: torch.device):
        self.device = device
        self.action_dim = action_dim
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.update_steps = 0

        self.target_network_countdown = config.target_network_countdown
        self.use_target_network = config.target_network_countdown > 0

        self.policy_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = SGBerD_wrapper(self.policy_net.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()
        self.config = config
        assert state_dict_equal(self.policy_net.state_dict(), self.target_net.state_dict())
        if config.path is not None:
            self.policy_net.load_state_dict(torch.load(config.path, map_location=self.device, weights_only=True))
            print("Weights loaded")
            assert not state_dict_equal(self.policy_net.state_dict(), self.target_net.state_dict())

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
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
            next_q = None
            if self.use_target_network:
                next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.policy_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * next_q
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def update_learning_rate(self, episode):
        lr = (self.config.lr - self.config.lr_lower) * (1 - (episode / self.config.episodes)) ** self.config.lr_decay_exponent + self.config.lr_lower
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update_noise(self):
        for group in self.optimizer.param_groups:
            group["magnitude"] = group["magnitude"] * NOISE_DECAY_FACTOR

    def update_target_network_countdown(self):
        if self.use_target_network:
            self.target_network_countdown -= 1
            if self.target_network_countdown <= 0:
                self.use_target_network = False
                print("Switched to using policy network for all future estimations.")


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

    agent = DDQNAgent(state_dim, action_dim, config, device)

    replay_buffer = ReplayBuffer(config.replay_size)
    replay_buffer_path = config.replay_buffer_path
    if replay_buffer_path is None and config.path is not None:
        replay_buffer_path = os.path.splitext(config.path)[0] + REPLAY_BUFFER_SUFFIX
    if replay_buffer_path is not None and os.path.exists(replay_buffer_path):
        replay_buffer = ReplayBuffer.load(replay_buffer_path)
        print(f"Replay buffer loaded from {replay_buffer_path}")
    epsilon = config.epsilon_start
    episode_rewards = []
    number_of_steps = []
    episode_loss = []
    training_losses_since_eval = []

    stats = []
    loss_count = 0

    reward_limit_count = 0
    epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    epsilon = max(config.epsilon_end, epsilon * (config.epsilon_decay**config.current_episode))
    for _ in range(int(config.current_episode/NOISE_UPDATE_FREQUENCY)):
        agent.update_noise()

        

    for episode in range(config.current_episode, config.episodes + 1):
        state, _ = env.reset(seed=(config.seed + episode) % MAX_SEED_VALUE)

        loss_list = []
        reward_list = []
        count = 0
        for _ in range(config.max_steps):
            count += 1
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            reward_list.append(reward)

            if len(replay_buffer) >= config.min_replay_size:
                loss = agent.train_step(replay_buffer)
                if loss is not None:
                    loss_list.append(loss)

            if done:
                break

        number_of_steps.append(count)
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        
        episode_rewards.append(reward_list)
        episode_loss.append(loss_list)
        if loss_list:
            episode_mean_loss = float(np.mean(loss_list))
            training_losses_since_eval.extend(loss_list)
        else:
            episode_mean_loss = None
        print(
            f"Episode {episode:4d}/{config.episodes} | "
            f"Reward: {sum(reward_list)} | "
            f"Loss: {episode_mean_loss if episode_mean_loss is not None else 'n/a'} | "
            f"Epsilon: {epsilon:6.3f}"
        )
        agent.update_target_network_countdown()
        if episode % 5 == 0:
            print("EVAL")

            eval_stats = evaluate(agent, config, device)
            eval_training_loss = (
                float(np.mean(training_losses_since_eval)) if training_losses_since_eval else None
            )
            stats.append(
                {
                    "episode": episode,
                    "evaluation": eval_stats,
                    "training_loss": eval_training_loss,
                }
            )
            training_losses_since_eval = []
            if eval_stats["reward"]["mean"] >= config.reward_limit:
                reward_limit_count += 1
                if reward_limit_count >= config.reward_limit_count:
                    print(
                        f"Reward limit reached {reward_limit_count} times: "
                        f"{eval_stats['reward']['mean']:.2f} >= {config.reward_limit:.2f}"
                    )
                    break
            else:
                reward_limit_count = 0
        if loss_list and np.mean(loss_list) < LOSS_MEAN_THRESHOLD:
            loss_count += 1
            if loss_count >= config.loss_threshold:
                agent.update_learning_rate(episode)
        else:
            loss_count = 0
        if episode % NOISE_UPDATE_FREQUENCY == 0:
            agent.update_noise()
        if episode >= config.save_after and episode <= config.save_before and episode % config.save_rate == 0:
            save(agent, replay_buffer, episode, config)

    env.close()
    return agent, device, stats

def save_res(trained_agent, stats, res, cfg, suffix=""):
    os.makedirs("output", exist_ok=True)
    run_name = f"{cfg.env_id}_seed_{cfg.seed}"
    filename = f"output/{run_name}_finished"
    if suffix:
        filename = f"{filename}_{suffix}"

    model_path = filename + ".pth"
    torch.save(trained_agent.policy_net.state_dict(), model_path)
    to_save = dataclasses.asdict(cfg)
    to_save["stats"] = stats
    to_save["res"] = res
    with open(filename + ".json", "w") as f:
        json.dump(to_save, f, indent=2)
        


def save(agent, replay_buffer: ReplayBuffer, episode, config):
    os.makedirs("models", exist_ok=True)
    run_name = f"{config.env_id}_seed_{config.seed}"
    filename = f"models/{run_name}_episode{episode}"

    to_save = dataclasses.asdict(config)
    to_save["current_episode"] = episode
    to_save["save_after"] = 0
    to_save["save_rate"] = 0
    to_save["save_before"] = 0
    to_save["target_network_countdown"] = 0
    to_save["path"] = filename + ".pth"
    to_save["replay_buffer_path"] = filename + REPLAY_BUFFER_SUFFIX
    print(to_save)
    with open(filename + ".json", "w") as f:
        json.dump(to_save, f, indent=2)
    torch.save(agent.policy_net.state_dict(), to_save["path"])
    replay_buffer.save(to_save["replay_buffer_path"])



def evaluate(agent: DDQNAgent, config: DDQNConfig, device: torch.device):
    env = gym.make(config.env_id)
    rewards = []
    steps = []
    for episode in range(1, config.eval_episodes + 1):
        eval_seed = (config.seed + config.eval_seed_offset + episode) % MAX_SEED_VALUE
        state, _ = env.reset(seed=eval_seed)
        total_reward = 0.0
        step_count = 0
        for _ in range(config.max_steps):
            step_count += 1
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(agent.policy_net(state_t).argmax(dim=1).item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        steps.append(step_count)
    env.close()
    print(f"[Eval] Mean reward over {config.eval_episodes} episodes: {np.mean(rewards):.2f}")
    return {
        "reward": {"mean": np.mean(rewards), "var": np.var(rewards), "list": rewards},
        "steps": {"mean": np.mean(steps), "var": np.var(steps), "list": steps},
    }


if __name__ == "__main__":
    cfg = parse_args()
    trained_agent, device, stats = train(cfg)
    evaluate(trained_agent, cfg, device)
