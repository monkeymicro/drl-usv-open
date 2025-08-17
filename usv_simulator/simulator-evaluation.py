
from dynamics_simulation import motionSim
import numpy as np
import random
import socket
import json
import traceback
from algo_struct.algo import rad_limit
from math import pow

def decode_data(data):
    recv_obser = json.loads(data)
    rspl = recv_obser['rspl']
    rspr = recv_obser['rspr']

    return rspl, rspr

def encode_data(trans):
    data = json.dumps(trans, sort_keys=True, indent=4, separators=(',', ':'))
    return data.encode('utf-8')

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

# motion sim
if __name__ == '__main__':
    boat_data = dict()

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
    
    my_mmg = motionSim.BoatMotionSim(delta_h=delta_h, name='usv')
    
    delta = 0
    try:
        x = np.array([0.0, 0.0, 0.0])
        u = np.array([0.0, 0.0, 0.0])
        V = 0
        last_u = 0.0

        delta = 0
        rsp = 0
        while True:
            sim_time += delta_h

            exp_speed = -1.5 * np.sin(sim_time/200*2*np.pi)
            exp_angle = 45 * np.sin(sim_time/200*2*np.pi)

            print(f'Expect speed and angle: {exp_speed}, {exp_angle}') 

            info, address = conn.recvfrom(1024)
            rspl, rspr = decode_data(info)
            print(f'Expect RPM: {rspl}, {rspr}')

            def rsp_to_force(rsp):
                a = 2.30549953e-04
                b = 3.88991253e-01
                c = -2.87920929e+02
                return a*rsp*abs(rsp) + b*rsp + c

            f_left = rsp_to_force(rspl)
            f_right = rsp_to_force(rspr)

            x, u, nu, acc = my_mmg.run_MMG_double_thrust(x=x, u=u, f_left=f_left, f_right=f_right)
            
            x[2] = rad_limit(x[2])
            
            V = u[0]/np.abs(u[0]) * np.sqrt(u[0]**2 + u[1]**2 + 1e-6)

            observation=dict()
            observation['exp_psi'] = exp_angle
            observation['exp_speed'] = exp_speed
            observation['rspl'] = rspl 
            observation['rspr'] = rspr 
            observation['psi'] = x[2]* 180.0/np.pi
            observation['v_r'] = u[2]* 180.0/np.pi 
            observation['v_x'] = u[0]
            observation['v_y'] = u[1]
            observation['V'] = V
            observation['acc_x'] = acc[0]
            observation['acc_y'] = acc[1]
            observation['acc_r'] = acc[2]
            observation['delta_psi'] = 180.0/np.pi * rad_limit(observation['exp_psi']*np.pi/180.0
                            - observation['psi']*np.pi/180.0)
            observation['delta_speed'] = observation['exp_speed'] - observation['V']
            
            boat_data['reward'] = get_reward(observation)
            print(f'observation is {observation}')
            
            if sim_time > all_sim_time:
                done = 1
            else:
                done = 0

            boat_data['observation'] = observation
            boat_data['terminated'] = done
            boat_data['truncated'] = done
            boat_data['time'] = sim_time
            data = encode_data(trans=boat_data)
            conn.send(data)

            if done:
                conn.close()
                socket_server.close()
                break

    except Exception:
        conn.close()
        socket_server.close()

        traceback.print_exc()
