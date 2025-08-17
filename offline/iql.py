# Inspired by:
# 1. implementation: https://github.com/gwthomas/IQL-PyTorch
# 2. paper: https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import argparse
from tqdm import trange
import pickle
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import socket
import json
TensorBatch = List[torch.Tensor]
HOME = os.path.dirname(os.path.realpath(__file__))

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = 'custom'  # OpenAI gym environment name
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 5  # How many episodes run during evaluation
    max_timesteps: int = int(5e5)  # Max time steps to run environment
    checkpoints_path: Optional[str] = f'{HOME}/models'  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 1_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize_states: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    max_action: float = 1.0
    port: int=11023
    name: str = "IQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        normalize_states: bool = True,
        normalize_reward: bool = False
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


# IQL
def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float, log_std_min: float = LOG_STD_MIN, log_std_max: float = LOG_STD_MAX):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)
        log_sigma = torch.clamp(log_sigma, self.log_std_min, self.log_std_max)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=True)[0].cpu().numpy()
        return action


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.v_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v_model(state)


class IQL:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        qf: Critic,
        qf_optimizer: torch.optim.Optimizer,
        vf: ValueFunction,
        vf_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        discount: float = 0.99,
        tau: float = 0.005,
        iql_deterministic: bool = False,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.qf = qf
        self.qf_target = copy.deepcopy(qf)
        self.qf_optimizer = qf_optimizer
        self.vf = vf
        self.vf_optimizer = vf_optimizer
        self.iql_deterministic = iql_deterministic

        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = batch

        log_dict = {}

        """
        Value function training
        """
        with torch.no_grad():
            q1, q2 = self.qf_target(state, action)
            q = torch.min(q1, q2).squeeze(1)

        v = self.vf(state).squeeze(1)
        vf_loss = asymmetric_l2_loss(q - v, self.iql_tau)

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        """
        Critic training
        """
        with torch.no_grad():
            v_next = self.vf(next_state).squeeze(1)
            target_q = reward.squeeze(1) + self.discount * (1.0 - done.squeeze(1)) * v_next

        q1, q2 = self.qf(state, action)
        qf_loss = F.mse_loss(q1.squeeze(1), target_q) + F.mse_loss(q2.squeeze(1), target_q)

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Actor training
        """
        with torch.no_grad():
            v = self.vf(state).squeeze(1)
            q1, q2 = self.qf(state, action)
            q = torch.min(q1, q2).squeeze(1)
            exp_adv = torch.exp(self.beta * (q - v)).clamp(max=EXP_ADV_MAX)

        policy_out, log_pi = self.actor(state, need_log_prob=True)
        # We use a deterministic actor, so log_pi is None here, and we can ignore it.
        if self.iql_deterministic:
            actor_loss = F.mse_loss(policy_out, action)
        else:
            actor_loss = torch.mean(exp_adv * log_pi) # In IQL, we minimize this loss.

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target
        soft_update(self.qf_target, self.qf, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "qf": self.qf.state_dict(),
            "vf": self.vf.state_dict(),
            "qf_target": self.qf_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "qf_optimizer": self.qf_optimizer.state_dict(),
            "vf_optimizer": self.vf_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.qf.load_state_dict(state_dict["qf"])
        self.vf.load_state_dict(state_dict["vf"])
        self.qf_target.load_state_dict(state_dict["qf_target"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.qf_optimizer.load_state_dict(state_dict["qf_optimizer"])
        self.vf_optimizer.load_state_dict(state_dict["vf_optimizer"])


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

    # rsp[0] = action[0]*5000
    # rsp[1] = action[1]*5000

    origin_data = {'boatname':'SLM7001',
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
    actor: nn.Module, device: str, n_episodes: int, action_dim: int = 2, tcp_socket: Optional[socket.socket] = None,
    state_mean: Optional[np.ndarray] = None, state_std: Optional[np.ndarray] = None
) -> np.ndarray:
    global rsp

    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        rsp = np.zeros(action_dim) # reset env
        action = np.random.uniform(-0.1, 0.1, size=action_dim)
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
            action = actor.act(torch.FloatTensor(state), device)
            action_data = encode_data(action, reset_flag=0)
            tcp_socket.send(action_data) # action
            info, addr = tcp_socket.recvfrom(1024)
            next_state, reward, terminated, truncated = decode_data(info) # next state

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
    if config.normalize_states:
        print("Normalize states enabled.")
    if config.normalize_reward:
        print("Normalize reward enabled.")

    state_dim = 14
    action_dim = 2
    rsp = np.zeros(action_dim) # reset env
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

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set up models
    actor = Actor(state_dim, action_dim, 256, config.max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    qf = Critic(state_dim, action_dim, 256).to(config.device)
    qf_optimizer = torch.optim.Adam(qf.parameters(), lr=config.qf_lr)
    vf = ValueFunction(state_dim, 256).to(config.device)
    vf_optimizer = torch.optim.Adam(vf.parameters(), lr=config.vf_lr)

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "qf": qf,
        "qf_optimizer": qf_optimizer,
        "vf": vf,
        "vf_optimizer": vf_optimizer,
        "iql_tau": config.iql_tau,
        "beta": config.beta,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        "iql_deterministic": config.iql_deterministic,
    }

    # Initialize policy
    trainer = IQL(**kwargs)

    print("---------------------------------------")
    print(f"Training IQL")
    print("---------------------------------------")

    evaluations = []
    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            # print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                action_dim=action_dim,
                tcp_socket=tcp_socket,
                state_mean=state_mean,
                state_std=state_std,
            )
            eval_score = eval_scores.mean()
            evaluations.append(eval_score)
            # print(f"Eval score: {eval_score}")

    if config.checkpoints_path is not None:
        torch.save(
            trainer.state_dict(),
            os.path.join(config.checkpoints_path, f"checkpoint.pt"),
        )
    with open(os.path.join(config.checkpoints_path, f"evaluations.pkl"), "wb") as f:
        pickle.dump(evaluations, f)


if __name__ == "__main__":
    train()