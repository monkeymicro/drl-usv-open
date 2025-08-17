from dynamics_simulation import motionSim
import numpy as np
import random
import socket
import json
import traceback
from algo_struct.algo import rad_limit
from math import pow

# recv from simulator
def decode_data(data):
    # load json
    recv_obser = json.loads(data)
    # print(recv_obser)
    rspl = recv_obser['rspl']
    rspr = recv_obser['rspr']

    return rspl, rspr

# send to simulator
def encode_data(trans):
    data = json.dumps(trans, sort_keys=True, indent=4, separators=(',', ':'))
    return data.encode('utf-8')

# calculate reward
def get_reward(observation: dict):
    score =100 - ( pow(180.0/np.pi * rad_limit((observation['exp_psi'] - observation['psi'])*np.pi/180.0) ,2)
                + 200 * pow(observation['exp_speed'] - observation['v_x'], 2)
                + observation['v_r']**2
                + 100 * observation['acc_x']**2
                + 100 * observation['acc_y']**2
                + 1e-7 * observation['rspl']**2
                + 1e-7 * observation['rspr']**2)

    print(f'score is {score}')
    return score

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=11023)
args = parser.parse_args()

# train policy
if __name__ == '__main__':
    boat_data = dict()
    # connect to simulator
    LOCAL_HOST = "127.0.0.1"
    LOCAL_PORT = args.port
    
    socket_server = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    socket_server.bind((LOCAL_HOST, LOCAL_PORT))
    socket_server.listen()
    conn, address = socket_server.accept()
    print(f"收到了客户端的连接，客户端信息是 {address}")

    all_sim_time = 200
    sim_time = 0
    delta_h = 0.2
    
    # usv motion simulator
    my_mmg = motionSim.BoatMotionSim(delta_h=delta_h, name='usv')
    
    delta = 0
    try:
        # expect reference direction and speed
        exp_speed = random.randint(-2, 2)
        exp_angle = random.randint(-90, 90)
        x = np.array([0.0, 0.0, 0.0])# 初始无人艇位置
        u = np.array([0.0, 0.0, 0.0])# 初始无人艇速度
        V = 0
        last_u = 0.0

        delta = 0
        rsp = 0
        while True:
            sim_time += delta_h
            print(f'expect speed and direction: {exp_speed}, {exp_angle}') 

            # 接收转速和舵角
            info, address = conn.recvfrom(1024)
            rspl, rspr = decode_data(info)
            print(f'thruster RPM: {rspl}, {rspr}')

            def rsp_to_force(rsp):
                a = 2.30549953e-04
                b = 3.88991253e-01
                c = -2.87920929e+02
                return a*rsp*abs(rsp) + b*rsp + c

            # rsp to force
            f_left = rsp_to_force(rspl)
            f_right = rsp_to_force(rspr)
            # update usv state
            x, u, nu, acc = my_mmg.run_MMG_double_thrust(x=x, u=u, f_left=f_left, f_right=f_right)
            
            x[2] = rad_limit(x[2]) # clip in [-pi, pi] 
            
            V = u[0]/np.abs(u[0]) * np.sqrt(u[0]**2 + u[1]**2 + 1e-6)
            
            # state
            observation=dict()
            observation['exp_psi'] = exp_angle # expected heading
            observation['exp_speed'] = exp_speed # expect surge speed m/s
            observation['rspl'] = rspl # port rpm
            observation['rspr'] = rspr # stpd rpm
            observation['psi'] = x[2]* 180.0/np.pi # heading angle degree
            observation['v_r'] = u[2]* 180.0/np.pi # z-axis rate degree/s
            observation['v_x'] = u[0] # x-axis speed
            observation['v_y'] = u[1] # y-axis speed
            observation['V'] = V # surge speed
            observation['acc_x'] = acc[0] # acceleration x-axis
            observation['acc_y'] = acc[1] # acceleration y-axis
            observation['acc_r'] = acc[2] # acceleration z-axis
            observation['delta_psi'] = 180.0/np.pi * rad_limit(observation['exp_psi']*np.pi/180.0 - observation['psi']*np.pi/180.0)# error heading
            observation['delta_speed'] = observation['exp_speed'] - observation['V']# error speed

            boat_data['reward'] = get_reward(observation)
            print(f'observation is {observation}')
            
            if sim_time > all_sim_time:
                exp_speed = random.randint(-2, 2)
                exp_angle = random.randint(-90, 90)
                x = np.array([0.0, 0.0, 0.0])# initial USV position
                u = np.array([0.0, 0.0, 0.0])# intial USV speed
                rspl = 0 
                rspr = 0 
                sim_time = 0
                done = 1
            else:
                done = 0

            boat_data['observation'] = observation
            boat_data['terminated'] = done
            boat_data['truncated'] = done
            data = encode_data(trans=boat_data)
            conn.send(data)

    except Exception:
        conn.close()
        socket_server.close()

        traceback.print_exc()
