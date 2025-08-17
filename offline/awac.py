# Inspired by:
# 1. paper: https://arxiv.org/abs/2006.09359
# 2. implementation: https://github.com/tinkoff-ai/CORL
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse

import pickle
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional
import socket
import json
from tqdm import trange

TensorBatch = List[torch.Tensor]
HOME = os.path.dirname(os.path.realpath(__file__))


@dataclass
class TrainConfig:
    name: str = "AWAC"
    checkpoints_path: Optional[str] = f'{HOME}/models'

    env_name: str = 'custom'
    deterministic_torch: bool = False
    device: str = "cuda"

    buffer_size: int = 1_000_000
    num_train_ops: int = 500_000
    batch_size: int = 256
    eval_frequency: int = 5000
    n_test_episodes: int = 5
    max_action: float = 1.0

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    awac_lambda: float = 0.3333
    port: int = 11025
    normalize_states: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        normalize_states: bool = True,
        normalize_reward: bool = False,
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

        self.normalize_reward_flag = normalize_reward
        if normalize_states:
            self._mean = torch.zeros(state_dim, dtype=torch.float32, device=device)
            self._std = torch.zeros(state_dim, dtype=torch.float32, device=device)
        if normalize_reward:
            self._reward_mean = torch.zeros(1, dtype=torch.float32, device=device)
            self._reward_std = torch.ones(1, dtype=torch.float32, device=device)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in minari, i.e. from Dict[str, np.array].
    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        if self.normalize_reward_flag:
            self._reward_mean = self._rewards.mean(dim=0, keepdim=True)
            self._reward_std = self._rewards.std(dim=0, keepdim=True) + 1e-3
            self._rewards = (self._rewards - self._reward_mean) / self._reward_std
            print("Rewards normalized.")

        print(f"Dataset size: {n_transitions}")

    def normalize_states(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes mean and std of states and normalizes them.
        """
        self._mean = self._states.mean(dim=0, keepdim=True)
        self._std = self._states.std(dim=0, keepdim=True) + eps
        self._states = (self._states - self._mean) / self._std
        self._next_states = (self._next_states - self._mean) / self._std
        return self._mean.cpu().numpy(), self._std.cpu().numpy()

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._min_action = min_action
        self._max_action = max_action

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        action = action_t[0].cpu().numpy()
        return action


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

    def _actor_loss(self, states, actions):
        with torch.no_grad():
            pi_action, _ = self._actor(states)
            v = torch.min(
                self._critic_1(states, pi_action), self._critic_2(states, pi_action)
            )

            q = torch.min(
                self._critic_1(states, actions), self._critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )

        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_actions, _ = self._actor(next_states)

            q_next = torch.min(
                self._target_critic_1(next_states, next_actions),
                self._target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self._gamma * (1.0 - dones) * q_next

        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)

        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)

        result = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self._actor.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._actor.load_state_dict(state_dict["actor"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"])


def decode_data(data):
    # load json
    recv_obser = json.loads(data)
    # print(recv_obser)
    observation = []
    observation_dict = recv_obser['observation']
    for key in observation_dict:
        observation.append(observation_dict[key])
    reward = recv_obser['reward']
    terminated = recv_obser['terminated']
    truncated = recv_obser['truncated']
    return observation, reward, terminated, truncated


def encode_data(action, reset_flag=1):
    global rsp
    rsp[0] = rsp[0] + action[0] * 500
    rsp[1] = rsp[1] + action[1] * 500
    rsp = np.clip(rsp, -5000, 5000)

    origin_data = {'boatname': 'SLM7001',
                   'restart': reset_flag,
                   'rudl': float(0),
                   'rudr': float(0),
                   'rspl': float(rsp[0]),
                   'rspr': float(rsp[1]),
                   'subSystem': "control"
                   }

    data = json.dumps(origin_data, sort_keys=True, indent=4, separators=(',', ':'))
    return data.encode('utf-8')


@torch.no_grad()
def eval_actor(
    actor: nn.Module,
    device: str,
    n_episodes: int,
    action_dim: int = 2,
    tcp_socket: Optional[socket.socket] = None,
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    global rsp

    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        rsp = np.zeros(action_dim)  # reset env
        action = np.zeros(action_dim, dtype=np.float32)
        action_data = encode_data(action, reset_flag=1)
        tcp_socket.send(action_data)
        # Initialize the environment and get its state
        info, addr = tcp_socket.recvfrom(1024)
        state, reward, terminated, truncated = decode_data(info)

        if state_mean is not None and state_std is not None:
            state = (np.asarray(state) - state_mean) / state_std
        state = np.squeeze(state)
        done = False
        episode_reward = 0.0
        while not done:
            # print("state: ", state)
            action = actor.act(torch.FloatTensor(state), device)
            action_data = encode_data(action, reset_flag=0)
            tcp_socket.send(action_data)  # action
            info, addr = tcp_socket.recvfrom(1024)
            next_state, reward, terminated, truncated = decode_data(info)  # next state

            if state_mean is not None and state_std is not None:
                next_state = (np.asarray(next_state) - state_mean) / state_std
            state = np.squeeze(next_state)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


@pyrallis.wrap()
def train(config: TrainConfig):
    state_dim = 14
    action_dim = 2
    rsp = np.zeros(action_dim)  # reset env
    # network socket
    REMOTE_HOST = '127.0.0.1'
    REMOTE_PORT = config.port
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((REMOTE_HOST, REMOTE_PORT))

    # 1. 加载 memory 数据
    with open(f'{HOME}/models/usv-dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    total_steps = dataset['obs'].shape[0]
    print('dataset total steps is :', total_steps)
    dataset = {
        "observations": dataset['obs'],
        "actions": dataset['acts'],
        "rewards": dataset['rews'],
        "next_observations": dataset['next_obs'],
        "terminals": dataset['dones'],
    }

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        total_steps,
        config.device,
        normalize_states=config.normalize_states,
        normalize_reward=config.normalize_reward
    )
    replay_buffer.load_dataset(dataset)

    if config.normalize_states:
        state_mean, state_std = replay_buffer.normalize_states()
    else:
        state_mean, state_std = None, None

    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    evaluations = []
    for t in trange(config.num_train_ops, ncols=80):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac.update(batch)
        if (t + 1) % config.eval_frequency == 0:
            eval_scores = eval_actor(
                actor,
                config.device,
                config.n_test_episodes,
                action_dim=action_dim,
                tcp_socket=tcp_socket,
                state_mean=state_mean,
                state_std=state_std,
            )

            eval_score = eval_scores.mean()
            evaluations.append(eval_score)
    if config.checkpoints_path is not None:
        torch.save(
            awac.state_dict(),
            os.path.join(config.checkpoints_path, f"checkpoint.pt"),
        )
    with open(os.path.join(config.checkpoints_path, f"evaluations.pkl"), "wb") as f:
        pickle.dump(evaluations, f)


if __name__ == "__main__":
    train()