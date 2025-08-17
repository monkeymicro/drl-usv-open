# Inspired by:
# 1. paper for SAC: https://arxiv.org/abs/1801.01290
# 2. implementation: https://github.com/MrSyee/pg-is-all-you-need
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import socket
import json
import os
import pickle
from copy import deepcopy
import math
from tqdm import trange

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, action_dim: int, size: int, batch_size: int = 32):
        """Initialize."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray,
        done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],)
    def save(self, path):
        with open(path, 'wb') as f:
            data = {
                'obs': self.obs_buf[:self.size],
                'acts': self.acts_buf[:self.size],
                'rews': self.rews_buf[:self.size],
                'next_obs': self.next_obs_buf[:self.size],
                'dones': self.done_buf[:self.size]
            }
            pickle.dump(data, f)

    def __len__(self) -> int:
        return self.size

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_num=64, max_action=1.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_num), nn.ReLU(),
            nn.Linear(hidden_num, hidden_num), nn.ReLU(),
            nn.Linear(hidden_num, hidden_num), nn.ReLU(), # Deeper Network
        )
        self.mu = nn.Linear(hidden_num, action_dim)
        self.log_std = nn.Linear(hidden_num, action_dim)
        self.max_action = max_action

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_std.bias, -1e-3, 1e-3)
    def forward(self, state, is_test = False):
        x = self.trunk(state)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -5, 2)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        # if train, deterministic is true, if test is false
        if is_test:
            action = mu # for test
        else:
            action = dist.rsample() # for train

        tanh_action, log_prob = torch.tanh(action), None

        if not is_test:
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6).sum(-1, keepdim=True)

        return tanh_action * self.max_action, log_prob

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
        return x @ self.weight + self.bias

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
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        q_values = self.critic(state_action).squeeze(-1)
        return q_values

class SAC:
    def __init__(self, obs_dim, action_dim, num_frames:int, random_steps:int, memory_size=100000, batch_size=256,
                q_num=10, hidden_num=64, gamma=0.99, tau=0.005, policy_update_freq=2,
                port: int=11023, ip: str='127.0.0.1'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_num = q_num
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_update_freq = policy_update_freq
        self.num_frames = num_frames
        self.random_steps = random_steps
        self.action_dim =  action_dim
        self.observation_dim = observation_dim

        self.actor = Actor(obs_dim, action_dim, hidden_num).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q_models = VectorizedCritic(
            self.observation_dim, self.action_dim, hidden_num, q_num
        )
        self.q_models.to(self.device)
        self.q_optims = torch.optim.Adam(
            self.q_models.parameters(), lr=3e-4
        )

        with torch.no_grad():
            self.target_q_models = deepcopy(self.q_models)

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim

        self.memory = ReplayBuffer(obs_dim, action_dim, memory_size, batch_size)
        self.total_step = 0

        # network, interact with boat.
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((ip, port))

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.random_steps:
            selected_action = np.random.uniform(-1, 1, size=self.action_dim)
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            )[0].detach().cpu().numpy()
            
        self.transition = [state, selected_action]
        
        return selected_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take an action and return the response of the env."""
        action_data = encode_data(action, reset_flag=0)
        self.tcp_socket.send(action_data)# action
        info, addr = self.tcp_socket.recvfrom(1024)
        next_state, reward, terminated, truncated = decode_data(info)# next state
        done = terminated or truncated

        self.transition += [reward, next_state, done]
        self.memory.store(*self.transition)

        return next_state, reward, done
    

    def update_model(self):
        samples = self.memory.sample_batch()
        s = torch.FloatTensor(samples['obs']).to(self.device)
        ns = torch.FloatTensor(samples['next_obs']).to(self.device)
        a = torch.FloatTensor(samples['acts']).to(self.device).requires_grad_(True)
        r = torch.FloatTensor(samples['rews']).unsqueeze(-1).to(self.device)
        d = torch.FloatTensor(samples['done']).unsqueeze(-1).to(self.device)

        # alpha update
        with torch.no_grad():
            _, log_prob = self.actor(s)

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().detach()

        # actor update
        if self.total_step % self.policy_update_freq == 0:
            new_action, log_prob = self.actor(s)
            q_vals = self.q_models(s, new_action)
            min_q = q_vals.min(0).values
            std_q = q_vals.std(0).mean().item()
            actor_loss = (self.alpha * log_prob - min_q).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        # critic update
        with torch.no_grad():
            next_a, next_logp = self.actor(ns)
            # print('next_logp shape is', next_logp.shape)
            q_targets = self.target_q_models(ns, next_a)
            min_q = q_targets.min(0).values.view(-1, 1)
            # print('min_q shape is', min_q.shape)
            # print('alpha shape is', self.alpha.shape)
            target_q = r + self.gamma * (1 - d) * (min_q - self.alpha * next_logp)
            # print('target_q shape is', target_q.shape)

        q_pred = self.q_models(s, a)
        # print('q_pred shape is', q_pred.shape)
        # q_loss = F.mse_loss(q_pred, target_q)
        q_loss = ((q_pred - target_q.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)
        # q_loss = q_loss.sum(dim=0)

        self.q_optims.zero_grad()
        q_loss.backward()
        self.q_optims.step()
        
        #  Target networks soft update
        with torch.no_grad():
            self.soft_update(self.target_q_models, self.q_models, tau=self.tau)
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

    def train(self):
        """Train the agent."""
        global rsp
        # env reset
        rsp = np.zeros(self.action_dim) # reset env
        action = np.random.uniform(-0.1, 0.1, size=self.action_dim)
        action_data = encode_data(action, reset_flag=1)
        # print(data)
        self.tcp_socket.send(action_data)
        # Initialize the environment and get its state
        info, addr = self.tcp_socket.recvfrom(1024)
        state, _, _, _= decode_data(info)

        score = 0
        frames = 0
        for self.total_step in trange(self.num_frames):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
            frames += 1
            # if episode ends
            if done:
                # print(f'frame is {self.total_step}, score is {score}, average reward is {score/frames}.')
                score_epoch.append(score)
                rsp = np.zeros(self.action_dim) # reset env
                action = np.random.uniform(-0.1, 0.1, size=self.action_dim)
                action_data = encode_data(action, reset_flag=1)
                self.tcp_socket.send(action_data)
                info, addr = self.tcp_socket.recvfrom(1024)
                observation, reward, terminated, truncated= decode_data(info)
                done = terminated or truncated
                state = observation
                score = 0
                frames = 0

            # if training is ready
            if (
                len(self.memory) >= self.batch_size 
                and self.total_step > self.random_steps
            ):
                self.update_model()

    def test(self, num_episodes: int = 10, memory_size: int = 10000):
        global rsp
        # memory for save
        buffer = ReplayBuffer(self.observation_dim, self.action_dim, size=memory_size)

        actor = Actor(state_dim=self.observation_dim, action_dim=self.action_dim, hidden_num=args.width).to(self.device)
        checkpoint = torch.load(f'{HOME}/models/{env_name}-{args.q_num}.pkl', map_location=self.device, weights_only=True)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        """Test the agent."""
        # global rsp
        # env reset
        rsp = np.zeros(self.action_dim) # reset env
        action = np.random.uniform(0, 0, size=self.action_dim)
        action_data = encode_data(action, reset_flag=1)
        # print(data)
        self.tcp_socket.send(action_data)
        # Initialize the environment and get its state
        info, addr = self.tcp_socket.recvfrom(1024)
        state, reward, terminated, truncated= decode_data(info)
        score = 0
        steps = 0
        frames = 0
        for i in range(num_episodes):
            done = False
            while not done:
                selected_action = actor(
                                torch.FloatTensor(state).to(self.device), is_test=True
                            )[0].detach().cpu().numpy()
                action_data = encode_data(selected_action, reset_flag=0)
                self.tcp_socket.send(action_data)# action
                info, addr = self.tcp_socket.recvfrom(1024)
                next_state, reward, terminated, truncated = decode_data(info)# next state
                done = terminated or truncated
                buffer.store(state, selected_action, reward, next_state, done)
                state = next_state
                score += reward
                steps += 1
                frames += 1

            print(f'Episode: {i}, fra mes: {frames}, Score : {score}, Average score: {score/frames}')
            rsp = np.zeros(self.action_dim) # reset env
            action = np.random.uniform(0, 0, size=self.action_dim)
            action_data = encode_data(action, reset_flag=1)
            self.tcp_socket.send(action_data)
            info, addr = self.tcp_socket.recvfrom(1024)
            state, reward, terminated, truncated= decode_data(info)
            done = terminated or truncated
            score = 0
            frames = 0

            if steps >= memory_size:
                break
        buffer.save(f'{HOME}/models/{env_name}-dataset.pkl')
        print(f"Memory saved")
def save_best_model(path):
    torch.save(obj={
        'actor_state_dict': agent.actor.state_dict(),
        'width' : args.width,
        'batch_size' : args.batch_size,
        'q_num' : args.q_num,
        'model': f'{env_name}',
        'score': score_epoch,
        }, f=path)

def decode_data(data):
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


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--q_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--port', type=int, default=11025)
parser.add_argument('--env_name', type=str, default='SAC')
args = parser.parse_args()

env_name = args.env_name
import random
HOME = os.path.dirname(os.path.realpath(__file__))
# parameters
num_frames = 500_000
memory_size = 500_000
initial_random_steps = 10_000
observation_dim = 14
action_dim = 2

rsp = np.zeros(action_dim) # reset env

# memory_demo = list()
score_epoch = []

agent = SAC(
    obs_dim=observation_dim,
    action_dim=action_dim,
    num_frames=num_frames,
    random_steps=initial_random_steps,
    memory_size=memory_size,
    batch_size=args.batch_size,
    q_num=args.q_num,
    hidden_num=args.width,
    port=args.port,
)

if True:  # train
    agent.train()
    save_best_model(f'{HOME}/models/{env_name}-{args.q_num}.pkl')
else:
    agent.test(num_episodes=1000, memory_size=1000_000)
    
print('Complete')
