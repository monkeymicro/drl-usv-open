from numpy import cos, sin, tan
import numpy as np
from disturbance import wave, wind
from algo_struct.myStruct import MyDynCoe
from collections import namedtuple
import os
def sec(angle: float) -> float:
    try:
        return 1 / cos(angle)
    except OSError as error_msg:
        print(error_msg)
        print("angle is zero!")


def uuv_b2n(u, x):
    phi = x[3]
    theta = x[4]
    psi = x[5]
    trans = np.array([
        [cos(psi) * cos(theta), cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi),
         cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi), 0, 0, 0],
        [sin(psi) * cos(theta), sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi),
         sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi), 0, 0, 0],
        [-sin(theta), cos(theta) * sin(phi), cos(phi) * cos(theta), 0, 0, 0],
        # [-sin(theta), cos(theta) * sin(phi), cos(phi) * cos(theta), 0, 0, 0],
        [0, 0, 0, 1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
        [0, 0, 0, 0, cos(phi), -sin(phi)],
        [0, 0, 0, 0, sin(phi) * sec(theta), cos(phi) * sec(theta)]
    ])

    return np.dot(trans, u)


def uuv_b2n_3dof(u, x):
    psi = x[2]
    trans = np.array([
        [cos(psi), - sin(psi), 0],
        [sin(psi), cos(psi), 0],
        [0, 0, 1]
    ])

    return np.dot(trans, u)


boat_status = namedtuple('Position', ('x', 'y', 'z', 'p', 'q', 'r'))
boat_velocity = namedtuple('Velocity', ('dx', 'dy', 'dz', 'dp', 'dq', 'dr'))

def unliner_kt(K, T, delta, alpha, deltam, x):
    dx1 = x[1]
    dx2 = (K*(deltam + delta) - x[1] - alpha* (x[1]**3))/T
    return np.array([dx1, dx2]) 

def deg_limit(deg):
	# 0~2pi to -pi~pi
	deg = deg - 360 * np.floor(deg / 360)
	if deg >= 180:
		deg = deg - 2 * 180
	if deg < -180:
		deg = deg + 2 * 180
	return deg

class BoatMotionSim():

    def __init__(self,
                delta_h: float,
                name: str) -> None:
        super().__init__()
        HOME = os.getcwd()
        print(HOME)

        boat_file = f'{HOME}/dynamics_simulation/boat.ini'
        
        self.boat = MyDynCoe()
        if name == 'usv':
            self.boat.boat_select(filename=boat_file, boat_name=name)#选择不同尺度的船
            # self.wave = wave.MyDynWave()
            # self.wave.boat_select(filename=boat_file,boat_name=myboat_name)
            self.wind = wind.myDynWind()
            self.wind.boat_select(filename=boat_file, boat_name=name)
            print('read USV settings.')
        elif name == 'rov':
            self.boat.rov_select(filename=boat_file, boat_name=name)
            print('read ROV settings.')
        else:
            print('cant recognize name.')
            
        self.h = delta_h

    # USV thruster and rudder model
    def mmg_equ(self,
                u: np.ndarray,
                x: np.ndarray,
                delta: float,
                f: float ):
        '''
        动力学方程
        :param u:
        :param x:
        :param delta:
        :param f:
        :return:
        '''
        if len(u) == 3 and len(x) == 3:
            psi = x[2]
            delta_r = delta

            uu = u[0]
            v = u[1]
            r = u[2]
            
            V = np.sqrt(uu**2 + v**2)

            m_matrix = np.array([
                [self.boat.m - 0.5*self.boat.rho_w*(self.boat.l **3) * self.boat.x_u_, 0, 0],
                [0, self.boat.m - 0.5*self.boat.rho_w*(self.boat.l **3) * self.boat.y_v_, -0.5*self.boat.rho_w*(self.boat.l **4) * self.boat.y_r_],
                [0, -0.5*self.boat.rho_w*(self.boat.l **4)* self.boat.n_v_, self.boat.ine_zz - 0.5*self.boat.rho_w*(self.boat.l **5) * self.boat.n_r_]
            ])
            
            c_matrix = np.array([
                [0.5*self.boat.rho_w*V*(self.boat.l **2) * self.boat.x_uu * np.abs(uu), 0, -self.boat.m * v],
                [0, 0.5*self.boat.rho_w*V*(self.boat.l **2)* self.boat.y_v, 0.5*self.boat.rho_w*V*(self.boat.l **3) * self.boat.y_r + self.boat.m * uu],
                [0, 0.5*self.boat.rho_w*V*(self.boat.l **3) * self.boat.n_v, 0.5*self.boat.rho_w*V*(self.boat.l **4) * self.boat.n_r * V]
            ])

            rudder = np.array([
                0,
                0,
                0.5*self.boat.rho_w*(self.boat.l **3) * self.boat.n_d_r * np.abs(uu)*uu * delta_r # 模拟waterjet
            ])

            tau_wind = self.wind.blendermann(u, psi)
            tau = np.array([
                f,
                0,
                0
            ])
            
            tau = f + tau_wind

            out = np.dot(np.linalg.inv(m_matrix), np.dot(c_matrix, u) + rudder + tau)
            return out

    # USV twin-waterjet model
    def mmg_equ_double_thrust(self,
                u: np.ndarray,
                x: np.ndarray,
                f_left: float,
                f_right: float ):
        '''
        动力学方程
        :param u:
        :param x:
        :param delta:
        :param f:
        :return:
        '''
        if len(u) == 3 and len(x) == 3:
            if u[2] > 30:
                u[2] = 30
            elif u[2] < -30:
                u[2] = -30
            V = np.sqrt(u[0]**2 + u[1]**2)

            m_matrix = np.array([
                [self.boat.m - 0.5*self.boat.rho_w*(self.boat.l **3) * self.boat.x_u_, 0, 0],
                [0, self.boat.m - 0.5*self.boat.rho_w*(self.boat.l **3) * self.boat.y_v_, -0.5*self.boat.rho_w*(self.boat.l **4) * self.boat.y_r_],
                [0, 0, self.boat.ine_zz - 0.5*self.boat.rho_w*(self.boat.l **5) * self.boat.n_r_]
            ])
            
            c_matrix = np.array([
                [0.5*self.boat.rho_w*V*(self.boat.l **2) * self.boat.x_uu * np.abs(u[0]), 0, -self.boat.m * u[1]],
                [0, 0.5*self.boat.rho_w*V*(self.boat.l **2)* self.boat.y_v, 0.5*self.boat.rho_w*V*(self.boat.l **3) * self.boat.y_r + self.boat.m * u[0]],
                [0, 0.5*self.boat.rho_w*V*(self.boat.l **3) * self.boat.n_v, 0.5*self.boat.rho_w*V*(self.boat.l **4) * self.boat.n_r]
            ])
            
            tau = np.array([
                f_left + f_right,
                0,
                1*f_left - 1*f_right,
            ])
            
            out = np.dot(np.linalg.inv(m_matrix), np.dot(c_matrix, u) + tau)
            return out
    
    # USV k-t model
    def kt(self,                 
            u: np.ndarray,
            x: np.ndarray,
            delta: float):

        temp = np.array([x[5], u[5]])
        
        k1 = unliner_kt(self.boat.K, self.boat.T, delta, self.boat.alpha, self.boat.deltam, temp)
        k1_ = temp + 0.5*self.h*k1
        k2 = unliner_kt(self.boat.K, self.boat.T, delta, self.boat.alpha, self.boat.deltam, k1_)
        k2_ = temp + 0.5*self.h*k2
        k3 = unliner_kt(self.boat.K, self.boat.T, delta, self.boat.alpha, self.boat.deltam, k2_)
        k3_ = temp + self.h*k3
        k4 = unliner_kt(self.boat.K, self.boat.T, delta, self.boat.alpha, self.boat.deltam, k3_)

        temp = temp + (self.h/6)*(k1 + 2*k2 + 2*k3 + k4)
        temp[0] = deg_limit(temp[0])
        
        return temp
        
    # ROV model multi-thrusters
    def ROV(self, x: np.ndarray, u: np.ndarray, v:np.ndarray) -> None:
        '''
        x - position
        u - velocity
        v - voltages signal \in [-1, 1]
        '''

        if len(u) == 6 and len(x) == 6 and len(v) == 6:
            M_RB = np.array([
                [self.boat.m, 0, 0, 0, 0, 0],
                [0, self.boat.m, 0, 0, 0, 0],
                [0, 0, self.boat.m, 0, 0, 0],
                [0, 0, 0, self.boat.ine_xx, 0, 0],
                [0, 0, 0, 0, self.boat.ine_yy, 0],
                [0, 0, 0, 0, 0, self.boat.ine_zz],
            ])
            M_A = np.array([
                [self.boat.x_u_, 0, 0, 0, 0, 0],
                [0, self.boat.y_v_, 0, 0, 0, 0],
                [0, 0, self.boat.z_w_, 0, 0, 0],
                [0, 0, 0, self.boat.ine_xx, 0, 0],
                [0, 0, 0, 0, self.boat.ine_yy, 0],
                [0, 0, 0, 0, 0, self.boat.ine_zz],
            ])
            m_matrix = M_RB + M_A
            
            C_RB = np.array([
                [0, 0, 0, 0 , self.boat.m*u[2], -self.boat.m*u[1]],
                [0, 0, 0, -self.boat.m*u[2], 0, self.boat.m*u[0]],
                [0, 0, 0, self.boat.m*u[1], -self.boat.m*u[0], 0],
                [0, self.boat.m* u[2], -self.boat.m * u[1], 0, -self.boat.ine_zz* u[5], -self.boat.ine_yy* u[4]],
                [-self.boat.m*u[2], 0, self.boat.m*u[0], self.boat.ine_zz* u[5], 0, self.boat.ine_xx* u[3]],
                [self.boat.m *u[1], -self.boat.m*u[0], 0, self.boat.ine_yy* u[4], -self.boat.ine_xx* u[3], 0],
            ])
            
            C_A = np.array([
                [0, 0, 0, 0, -self.boat.z_w_*u[2], self.boat.y_v_*u[1]],
                [0, 0, 0, self.boat.z_w_*u[2], 0, -self.boat.x_u_*u[0]],
                [0, 0, 0, -self.boat.y_v_*u[1], self.boat.x_u_*u[0], 0],
                [0, -self.boat.z_w_*u[2], self.boat.y_v_*u[1], 0, -self.boat.n_r_*u[5], self.boat.m_q_*u[4]],
                [self.boat.z_w_*u[2], 0, -self.boat.x_u_*u[0], self.boat.n_r_*u[5], 0, -self.boat.k_p_*u[3]],
                [-self.boat.y_v_*u[1], self.boat.x_u_*u[0], 0, -self.boat.m_q_ * u[4], self.boat.k_p_ * u[3], 0],
            ])
            
            c_matrix = C_RB + C_A
            
            d_matrix = np.array([
                [-self.boat.x_u, 0, 0, 0, 0, 0],
                [0, -self.boat.y_v, 0, 0, 0, 0],
                [0, 0, -self.boat.z_w, 0, 0, 0],
                [0, 0, 0, -self.boat.k_p, 0, 0],
                [0, 0, 0, 0, -self.boat.m_q, 0],
                [0, 0, 0, 0, 0, -self.boat.n_r],
            ])
            
            d_n_matrix = np.array([
                [-self.boat.x_uu*abs(u[0]), 0, 0, 0, 0, 0],
                [0, -self.boat.y_vv*abs(u[1]), 0, 0, 0, 0],
                [0, 0, -self.boat.z_ww*abs(u[2]), 0, 0, 0],
                [0, 0, 0, -self.boat.k_pp*abs(u[3]), 0, 0],
                [0, 0, 0, 0, -self.boat.m_qq*abs(u[4]), 0],
                [0, 0, 0, 0, 0, -self.boat.n_rr*abs(u[5])],
            ])
            
            D_m = d_matrix + d_n_matrix 
            
            if x[2] < self.boat.H/2.0:
                Delta = self.boat.Delta * (1- (self.boat.H/2.0 - x[2])/self.boat.H)
            else:
                Delta = self.boat.Delta
            
            W = self.boat.m * self.boat.g_e
            B = W
            g_matrix = np.array([
                (W-B)* np.sin(x[4]),
                -(W-B)* np.cos(x[4])* np.sin(x[3]),
                (W-B)* np.cos(x[4])* np.cos(x[3]),
                self.boat.y_b * B * np.cos(x[4])*np.cos(x[3]) - self.boat.z_b * B * np.cos(x[4])*np.sin(x[3]),
                -self.boat.z_b * B * np.sin(x[4]) - self.boat.x_b * B * np.cos(x[4])*np.sin(x[3]),
                self.boat.x_b * B * np.cos(x[4])*np.cos(x[3]) + self.boat.y_b * B * np.sin(x[4])
            ])
            
            thrust = (-140.3 * v**9) + (389.9 * v**7) - (404.1 * v**5) + (176.0 * v**3) + (8.9 * v)
            print(f'thrust is {thrust}')
            attached = 45.0 * np.pi/180.0
            ly = self.boat.L /2.0
            lx = self.boat.B / 2.0
            
            tau = np.array([
                [-np.cos(attached), -np.cos(attached), np.cos(attached), np.cos(attached), 0, 0],
                [-np.sin(attached), np.sin(attached), -np.sin(attached), np.sin(attached), 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, -ly, ly],
                [0, 0, 0, 0, 0, 0],
                [-np.cos(attached)*ly - np.sin(attached)*lx, np.cos(attached)*ly + np.sin(attached)*lx, np.cos(attached)*ly + np.cos(attached)*lx, -np.cos(attached)*ly - np.sin(attached)*lx, 0, 0],
            ])
            print(np.dot(tau, thrust))
            
            out = np.dot(np.linalg.inv(m_matrix), np.dot(-c_matrix, u) + np.dot(-D_m, u) - g_matrix + np.dot(tau, thrust))
            
            return out
    
    # ODE Runge-Kutta method
    def runMMG(self, x: np.ndarray, u: np.ndarray, delta:float, f:float) -> None:
        k1 = self.mmg_equ(u=u, x=x, delta=delta, f=f)
        k2 = self.mmg_equ(u=u + 0.5 * self.h * k1, x=x, delta =delta, f=f)
        k3 = self.mmg_equ(u=u + 0.5 * self.h * k2, x=x, delta =delta, f=f)
        k4 = self.mmg_equ(u=u + self.h * k3, x=x, delta=delta, f=f)
        acc = 1.0/6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        u = u + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        origin_u = u
        nu = uuv_b2n_3dof(u, x)
        x += nu*self.h

        return x, origin_u, nu, acc

    def run_MMG_double_thrust(self, x: np.ndarray, u: np.ndarray, f_left:float, f_right:float) -> None:
        k1 = self.mmg_equ_double_thrust(u=u, x=x, f_left=f_left, f_right=f_right)
        k2 = self.mmg_equ_double_thrust(u=u + 0.5 * self.h * k1, x=x, f_left =f_left, f_right=f_right)
        k3 = self.mmg_equ_double_thrust(u=u + 0.5 * self.h * k2, x=x, f_left =f_left, f_right=f_right)
        k4 = self.mmg_equ_double_thrust(u=u + self.h * k3, x=x, f_left=f_left, f_right=f_right)
        acc = 1.0/6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        u = u + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        origin_u = u
        nu = uuv_b2n_3dof(u, x)
        x += nu*self.h

        return x, origin_u, nu, acc

    def run_ROV(self, x: np.ndarray, u: np.ndarray, v:np.ndarray) -> None:
        k1 = self.ROV(u=u, x=x, v=v)
        k2 = self.ROV(u=u + 0.5 * self.h * k1, x=x, v=v)
        k3 = self.ROV(u=u + 0.5 * self.h * k2, x=x, v=v)
        k4 = self.ROV(u=u + self.h * k3, x=x, v=v)
        acc = 1.0/6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        u = u + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        origin_u = u
        nu = uuv_b2n(u, x)
        x += nu*self.h
        
        print(f'x is {x}')
        print(f'u is {u}')

        return x, origin_u, nu, acc
    
    def runKt(self, x: np.ndarray, u: np.ndarray, delta:float, f:float):

        temp = self.boat.Kt(u=u, x=x, delta=delta)
        origin_u = f*0.005
        u[0] = origin_u*np.sin(temp[0] * np.pi/180.0)
        u[1] = origin_u*np.cos(temp[0] * np.pi/180.0)
        u[5] = temp[1]
        x += u*self.h
        x[5] = temp[0]
        
        print('x:', x)
        return x, u, origin_u
    
    