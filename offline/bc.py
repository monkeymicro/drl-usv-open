# Inspired by:
# implementation: https://github.com/sfujim/TD3_BC
# paper: https://arxiv.org/pdf/2106.06860.pdf
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import trange
import pickle
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import json
TensorBatch = List[torch.Tensor]

HOME = os.path.dirname(os.path.realpath(__file__))

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 5  # How many episodes run during evaluation
    max_timesteps: int = int(5e5)  # Max time steps to run environment
    checkpoints_path: Optional[str] = f'{HOME}/models'  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    # BC
    buffer_size: int = 1_000_000  # Replay buffer size
    frac: float = 0.1  # Best data fraction to use
    max_traj_len: int = 1000  # Max trajectory length
    normalize_states: bool = True  # Normalize states
    normalize_reward: bool = False # Normalize rewards
    env: str = "custom"
    name: str = "BC"
    port: int = 11023

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
            # print("Rewards normalized.")

        # print(f"Dataset size: {n_transitions}")

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
        # print(data)
        tcp_socket.send(action_data)
        # Initialize the environment and get its state
        info, addr = tcp_socket.recvfrom(1024)
        state, reward, terminated, truncated= decode_data(info)
        
        if state_mean is not None and state_std is not None:
            state = (np.asarray(state) - state_mean) / state_std

        done = False
        episode_reward = 0.0
        while not done:
            action = actor.act(torch.FloatTensor(state), device)
            action_data = encode_data(action, reset_flag=0)
            tcp_socket.send(action_data)# action
            info, addr = tcp_socket.recvfrom(1024)
            next_state, reward, terminated, truncated = decode_data(info)# next state
            
            if state_mean is not None and state_std is not None:
                next_state = (np.asarray(next_state) - state_mean) / state_std
            
            state = next_state
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def keep_best_trajectories(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
):
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(frac * len(sort_ord)))]

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)
    dataset["observations"] = dataset["observations"][order]
    dataset["actions"] = dataset["actions"][order]
    dataset["next_observations"] = dataset["next_observations"][order]
    dataset["rewards"] = dataset["rewards"][order]
    dataset["terminals"] = dataset["terminals"][order]


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float=1.0):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class BC:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
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
    keep_best_trajectories(dataset, config.frac, config.discount)

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

    actor = Actor(state_dim, action_dim).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "device": config.device,
    }
    # Initialize policy
    trainer = BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

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

    if config.checkpoints_path is not None:
        torch.save(
            trainer.state_dict(),
            os.path.join(config.checkpoints_path, f"checkpoint.pt"),
        )
    with open(os.path.join(config.checkpoints_path, f"evaluations.pkl"), "wb") as f:
        pickle.dump(evaluations, f)


if __name__ == "__main__":
    train()