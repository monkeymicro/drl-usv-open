# Inspired by:
# 1. paper: https://arxiv.org/abs/1802.09477
# 2. implementation: https://github.com/sfujim/TD3
# 3. implementation: https://github.com/MrSyee/pg-is-all-you-need
# 4. implementation: https://github.com/openai/spinningup
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
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_num: int =64):
        """Initialize."""
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, out_dim),
            nn.Tanh(),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_num: int =64):
        """Initialize."""
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class TD3Agent:
    """TD3Agent interacting with environment.
    """

    def __init__(
        self,
        num_frames: int,
        memory_size: int,
        batch_size: int,
        gamma: float = 0.99,
        tau: float = 5e-3,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        target_policy_noise_clip: float = 0.5,
        random_steps: int = int(1e4),
        policy_update_freq: int = 2,
        obs_dim: int=4,
        action_dim: int=1,
        q_num: int=2,
        hidden_num: int=256,
        port: int=11023,
        ip: str = '127.0.0.1',
    ):
        """Initialize."""
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_noise = target_policy_noise
        self.noise_clip = target_policy_noise_clip
        self.exploration_noise = exploration_noise
        self.memory = ReplayBuffer(obs_dim, action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.random_steps = random_steps
        self.policy_update_freq = policy_update_freq
        self.num_frames = num_frames
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        lr_actor = 3e-4
        lr_critic = 3e-4
        # noise
        
        # networks
        self.actor = Actor(obs_dim, action_dim, hidden_num).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, hidden_num).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q_num = q_num
        # q function, N for overestimation
        self.q_model = [Critic(obs_dim + action_dim, hidden_num=hidden_num).to(self.device) for _ in range(q_num)]

        self.q_model_target = [Critic(obs_dim + action_dim, hidden_num=hidden_num).to(self.device) for _ in range(q_num)]
        for i in range(q_num):
            self.q_model_target[i].load_state_dict(self.q_model[i].state_dict())

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.q_model_optim = [optim.Adam(self.q_model[i].parameters(), lr=lr_critic) for i in range(q_num)]

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0
        # network, interact with boat.
        self.TCP_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.TCP_socket.connect((ip, port))
 
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.random_steps:
            selected_action = np.random.uniform(-1, 1, size=self.action_dim)
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()
        selected_action += np.random.normal(0, self.exploration_noise, selected_action.size)
        selected_action = np.clip(selected_action, -1, 1)
        self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        # print(f'step action is {action}')
        action_data = encode_data(action, reset_flag=0)
        self.TCP_socket.send(action_data)# action
        info, addr = self.TCP_socket.recvfrom(1024)
        observation, reward, terminated, truncated = decode_data(info)# next state
        done = terminated or truncated

        self.transition += [reward, observation, done]
        self.memory.store(*self.transition)
    
        return observation, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        states = torch.FloatTensor(samples["obs"]).to(device)
        next_states = torch.FloatTensor(samples["next_obs"]).to(device)
        actions = torch.FloatTensor(samples["acts"]).to(device)
        rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        masks = 1 - dones
        
        # Select action according to actor and add clipped noise
        noise = (torch.randn_like(actions) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )

        next_actions = (self.actor_target(next_states) + noise).clamp(
            -1.0, 1.0
        )
        
        q_pred_l = list()
        for i in range(len(self.q_model)):
            q_pred_l.append(self.q_model[i](states, actions))
            
        # min (Q_1', Q_2')
        for i in range(self.q_num):
            if i == 0:
                q_pred = self.q_model[i](next_states, next_actions) # s', a'
            else:
                q_pred = torch.min(q_pred, self.q_model[i](next_states, next_actions))

        q_target = rewards + self.gamma * q_pred * masks

        # critic loss
        q_loss_l = list()
        for i in range(len(self.q_model)):
            q_loss_l.append( F.mse_loss(q_pred_l[i], q_target.detach()))

        # train Q functions. Don't sum loss
        for i in range(self.q_num):
            self.q_model_optim[i].zero_grad()
            q_loss_l[i].backward()
            self.q_model_optim[i].step()

        if self.total_step % self.policy_update_freq == 0:
            # train actor
            next_actions_actor = self.actor(states)
            q_values = self.q_model[0](states, next_actions_actor)
            actor_loss = -q_values.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update
            self._target_soft_update()

    def train(self):
        """Train the agent."""
        global rsp
        action = np.random.uniform(0, 0, size=self.action_dim)
        action_data = encode_data(action, reset_flag=1)
        # print(data)
        self.TCP_socket.send(action_data)

        # Initialize the environment and get its state
        info, addr = self.TCP_socket.recvfrom(1024)
        state, reward, terminated, truncated= decode_data(info)
        score = 0
        i_episode = 0
        frame = 0
        for self.total_step in trange(self.num_frames):
            frame += 1.0
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

            # if episode ends
            if done:
                rsp = np.array([0.0, 0.0])
                i_episode += 1
                record_score.append(score)
                # print(f'episode is {i_episode}, score is {score/frame}') 

                action = np.random.uniform(0, 0, size=self.action_dim)
                action_data = encode_data(action, reset_flag=1)
                # print(data)
                self.TCP_socket.send(action_data)
                # Initialize the environment and get its state
                info, addr = self.TCP_socket.recvfrom(1024)
                state, reward, terminated, truncated= decode_data(info)
                done = terminated or truncated
                score = 0
                frame = 0

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
        checkpoint = torch.load(f'{HOME}/models/{env_name}.pkl', map_location=self.device, weights_only=True)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        """Test the agent."""
        # global rsp
        # env reset
        rsp = np.zeros(self.action_dim) # reset env
        action = np.random.uniform(-0.1, 0.1, size=self.action_dim)
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
                                torch.FloatTensor(state).to(self.device)
                            ).detach().cpu().numpy()
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

            print(f'Episode: {i}, frames: {frames}, Score : {score}, Average score: {score/frames}')
            rsp = np.zeros(self.action_dim) # reset env
            action = np.random.uniform(-0.1, 0.1, size=self.action_dim)
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

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for i in range(self.q_num):
            for t_param, l_param in zip(
                self.q_model_target[i].parameters(), self.q_model[i].parameters()
            ):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


def save_best_model(path):
    torch.save(obj={
        'actor_state_dict': agent.actor.state_dict(),
        'width' : args.width,
        'batch_size' : args.batch_size,
        'q_num' : args.q_num,
        'model': f'{env_name}',
        'score': record_score,
        }, f=path)


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


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--q_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--port', type=int, default=11023)
parser.add_argument('--env_name', type=str, default='TD3')
args = parser.parse_args()
env_name = args.env_name
# parameters
num_frames = 500_000
memory_size = 500_000
initial_random_steps = 10_000

import random

rsp = np.array([0.0, 0.0]) # reset action

HOME = os.getcwd()
record_score = []

agent = TD3Agent(
    num_frames=num_frames,
    memory_size=memory_size,
    batch_size=args.batch_size,
    random_steps=initial_random_steps,
    obs_dim=14,
    action_dim=2,
    q_num = args.q_num,
    hidden_num=args.width,
    port = args.port,
)

if True:  # train
    agent.train()
    save_best_model(f'{HOME}/models/{env_name}.pkl')
else:
    # for producing dataset
    agent.test(num_episodes=1000, memory_size=1000_000)
    
print('Complete')