# Inspired by:
# 1. paper for REM: https://arxiv.org/abs/1907.04543
# 2. implementation: https://github.com/google-research/batch_rl
import math
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import trange
import pickle
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import socket
import json
from torch.distributions import Normal

HOME = os.path.dirname(os.path.realpath(__file__))

@dataclass
class TrainConfig:
    name: str = "REM"
    # model params
    hidden_dim: int = 256
    num_critics: int = 50
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 1_000_000
    env_name: str = 'custom'
    batch_size: int = 256
    num_epochs: int = 500
    num_updates_on_epoch: int = 1000
    normalize_states: bool = True
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 5
    eval_every: int = 5
    # general params
    checkpoints_path: Optional[str] = f'{HOME}/models'
    port:int = 11023
    device: str = "cuda" # "cpu"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
TensorBatch = List[torch.Tensor]


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

# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
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

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = True
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)
        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = state.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        action = action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        sa = torch.cat([state, action], dim=2)
        q_ensemble = self.critic(sa)
        return q_ensemble


class REM:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critics: VectorizedCritic,
        critics_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_learning_rate: float = 1e-4,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critics = critics
        self.critics_target = deepcopy(critics)
        self.critics_optimizer = critics_optimizer

        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.num_critics = self.critics.num_critics

        # automatic entropy tuning
        self.target_entropy = -torch.prod(
            torch.Tensor(self.actor.action_dim).to(self.device)
        ).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_learning_rate
        )
        self.alpha = self.log_alpha.exp().item()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = batch

        log_dict = {}

        """
        Alpha training
        """
        with torch.no_grad():
            _, log_prob = self.actor(state, need_log_prob=True)
        alpha_loss = (
            -self.log_alpha * (log_prob.detach() + self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()
        log_dict["alpha"] = self.alpha
        log_dict["alpha_loss"] = alpha_loss.item()

        """
        Actor training
        """
        # We need to re-sample actions to stop gradients from flowing into the actor
        action_new, log_prob = self.actor(state, need_log_prob=True)
        q_ensemble = self.critics(state, action_new)

        actor_loss = (self.alpha * log_prob - q_ensemble.mean(dim=0)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        log_dict["actor_loss"] = actor_loss.item()

        """
        Critic training
        """
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state, need_log_prob=True)
            # compute target Q
            target_q = self.critics_target(next_state, next_action)
            # take min over the first num_critics - 1 critics
            q_target_values = torch.mean(
                target_q + self.alpha * next_log_prob.unsqueeze(0),
                dim=0,
                keepdim=True
            ).min(dim=0)[0]
            target_q = reward + (1.0 - done) * self.gamma * q_target_values
            target_q = target_q.squeeze(0)

        q_ensemble = self.critics(state, action)
        q_loss = (q_ensemble - target_q.unsqueeze(0)).pow(2).mean()

        self.critics_optimizer.zero_grad()
        q_loss.backward()
        self.critics_optimizer.step()

        log_dict["q_loss"] = q_loss.item()

        # update target
        soft_update(self.critics_target, self.critics, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critics": self.critics.state_dict(),
            "critics_optimizer": self.critics_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critics.load_state_dict(state_dict["critics"])
        self.critics_target = deepcopy(self.critics)
        self.critics_optimizer.load_state_dict(state_dict["critics_optimizer"])
        self.log_alpha.data = state_dict["log_alpha"].data
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        self.alpha = self.log_alpha.exp().item()

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
        state = np.squeeze(state)
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

    actor = Actor(state_dim, action_dim, config.hidden_dim, config.max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)

    critics = VectorizedCritic(state_dim, action_dim, config.hidden_dim, config.num_critics).to(config.device)
    critics_optimizer = torch.optim.Adam(critics.parameters(), lr=config.critic_learning_rate)

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critics": critics,
        "critics_optimizer": critics_optimizer,
        "gamma": config.gamma,
        "tau": config.tau,
        "alpha_learning_rate": config.alpha_learning_rate,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training REM")
    print("---------------------------------------")

    # Initialize policy
    trainer = REM(**kwargs)

    evaluations = []
    for t in trange(int(config.num_epochs)):
        for i in range(config.num_updates_on_epoch):
            batch = replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.train(batch)

        # Evaluate episode
        if (t + 1) % config.eval_every == 0:
            # print(f"Epoch: {t + 1}, timesteps: {(t+1)*config.num_updates_on_epoch}")
            eval_scores = eval_actor(
                actor,
                device=config.device,
                n_episodes=config.eval_episodes,
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