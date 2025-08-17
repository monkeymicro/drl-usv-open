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

def test(observation_dim: int, action_dim:int, device: str, tcp_socket) -> None:
    """Test the agent."""
    global rsp

    actor = Actor(in_dim=observation_dim, out_dim=action_dim, hidden_num=args.width).to(device)
    checkpoint = torch.load(f'{HOME}/models/{env_name}.pkl', map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    rsp = np.zeros(action_dim) # reset env
    action = np.random.uniform(0, 0, size=action_dim)
    action_data = encode_data(action, reset_flag=1)
    # print(data)
    tcp_socket.send(action_data)
    # Initialize the environment and get its state
    info, addr = tcp_socket.recvfrom(1024)
    state, reward, terminated, truncated= decode_data(info)
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
        done = terminated or truncated
        state = next_state
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
parser.add_argument('--q_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--port', type=int, default=11023)
parser.add_argument('--env_name', type=str, default='TD3')
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