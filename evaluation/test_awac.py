import copy
import os
import random
from typing import Dict, List, Tuple
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, pickle, json, socket
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # Normalize rewards if flag is set
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
        action = policy.mean
        # action.clamp_(self._min_action, self._max_action)
        # log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action

def test(observation_dim: int, action_dim:int, device: str, tcp_socket) -> None:
    """Test the agent."""
    
    # load dataset
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
        observation_dim,
        action_dim,
        total_steps,
        device,
        normalize_states=True,
        normalize_reward=False,
    )
    replay_buffer.load_dataset(dataset)
    state_mean, state_std = replay_buffer.normalize_states()

    global rsp

    # 创建 actor 网络
    actor = Actor(state_dim=observation_dim, action_dim=action_dim, hidden_dim=args.width).to(device)
    checkpoint = torch.load(f'{HOME}/models/{env_name}/checkpoint.pt', map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    rsp = np.zeros(action_dim) # reset env
    action = np.zeros(action_dim)
    action_data = encode_data(action, reset_flag=1)
    # print(data)
    tcp_socket.send(action_data)
    # Initialize the environment and get its state
    info, addr = tcp_socket.recvfrom(1024)
    state, reward, terminated, truncated= decode_data(info)
    # Apply normalization if stats are provided
    if state_mean is not None and state_std is not None:
        state = (np.asarray(state) - state_mean) / state_std
    state = np.squeeze(state)
    score = 0
    steps = 0
    frames = 0

    done = False
    while not done:
        selected_action = actor(
                        torch.FloatTensor(state).to(device)
                    ).detach().cpu().numpy()
        action_data = encode_data(selected_action, reset_flag=0)
        tcp_socket.send(action_data)# action
        info, addr = tcp_socket.recvfrom(1024)
        next_state, reward, terminated, truncated = decode_data(info)# next state
        # Apply normalization if stats are provided
        if state_mean is not None and state_std is not None:
            next_state = (np.asarray(next_state) - state_mean) / state_std
        state = np.squeeze(next_state)
        done = terminated or truncated
        score += reward
        steps += 1
        frames += 1

    print(f'frames: {frames}, Score : {score}, Average score: {score/frames}')
        
    with open(f'{HOME}/results/{env_name}_test_data.pkl', 'wb') as f:
        pickle.dump(save_data, f)
def decode_data(data):
    # load json
    recv_obser = json.loads(data)
    # print(recv_obser)
    observation = []
    observation_dict = recv_obser['observation']
    for key in observation_dict:
        observation.append(observation_dict[key])
        save_data[key].append(observation_dict[key])
    save_data['time'].append(recv_obser['time'])
    reward = recv_obser['reward']
    terminated = recv_obser['terminated']
    truncated = recv_obser['truncated']
    return observation, reward, terminated, truncated

def encode_data(action: np.ndarray, reset_flag=1):
    global rsp
    
    rsp[0] = rsp[0] + action[0] * 500
    rsp[1] = rsp[1] + action[1] * 500
    rsp = np.clip(rsp, -5000, 5000)
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

import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--port', type=int, default=11023)
parser.add_argument('--env_name', type=str, default= 'AWAC-custom')
args = parser.parse_args()
env_name = args.env_name

obs_dim = 14
action_dim = 2

# device: cpu / gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# network, interact with boat.
TCP_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
TCP_socket.connect(('127.0.0.1', args.port))

rsp = np.array([0.0, 0.0]) # reset action

HOME = os.getcwd()
save_data = dict()
save_data['exp_psi'] = []
save_data['exp_speed'] = []
save_data['rspl'] = []
save_data['rspr'] = []
save_data['psi'] = []
save_data['V'] = [] 
save_data['v_r'] = []
save_data['v_x'] = []
save_data['v_y'] = []
save_data['V'] = []
save_data['acc_x'] = []
save_data['acc_y'] = []
save_data['acc_r'] = []
save_data['delta_psi'] = []
save_data['delta_speed'] = []
save_data['time'] = []

test(observation_dim=obs_dim, action_dim=action_dim, device=device, tcp_socket=TCP_socket)

print('Complete')