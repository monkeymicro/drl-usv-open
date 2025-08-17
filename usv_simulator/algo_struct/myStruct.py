import configparser
def readIni(filename):
	config = configparser.ConfigParser()
	config.read(filename, encoding='utf-8')
	return config
class MyDynCoe:
	def __init__(self):
		
		self.rho_w = 1000.0
		self.g_e = 9.81
		self.x_g = 0.0
		self.y_g = 0.0
		self.z_g = 0.0
		self.x_b = 0.0
		self.y_b = 0.0
		self.z_b = 0.01
		self.l = 2.38 # length
		self.b = 1.0

		self.m = 127.0 # mass
		self.Delta = self.m / self.rho_w # T
		self.ine_xx = 15.73
		self.ine_xy = 0.0
		self.ine_yy = 70.27
		self.ine_zz = 71.38
		self.ine_xz = 0.0
		self.ine_yz = 0.0

		# hydrodynamic coefficient
		self.x_u_ = -4.8e-3
		self.x_u = 0
		self.x_uu = -9.3e-3
		self.y_r_ = -9.67e-4
		self.y_v_ = -3.66e-2
		self.y_r = -4.8e-3
		self.y_v = -5.69e-2
		self.y_vv = 0
		self.y_d_r = -2.15e-2
		self.z_q_ = -1.86e-4
		self.z_w_ = -6.8e-2 
		self.z_w = -8.3e-3 
		self.z_ww = 0
		self.z_q = 7.61e-3

		# self.z_d_s = -3.3e-2
		self.z_d_s = .0
		self.k_p_ = 0
		self.k_p = 0
		self.k_pp = 0.0
		self.k_vq = 0.0
		self.k_wr = 0.0
		self.m_q_ = -4.1e-4
		self.m_w_ = -1.82e-4
		self.m_q = -2.07e-3
		self.m_qq = .0
		self.m_w = .0
		self.m_d_s = 2.13e-3
		self.n_r_ = -2.4e-3
		self.n_v_ = 9.54e-4
		self.n_r = -0.0664
		self.n_rr= -0.0054
		self.n_v = -1.86e-2
		self.n_vv = 0.0
		self.n_d_r = 0.0829

	def boat_select(self, filename, boat_name):
		config = readIni(filename)
		print(f'ini file is {filename}')
		self.rho_w = float(config.get(boat_name, 'rho'))
		self.g_e = float(config.get(boat_name, 'g'))
		self.l = float(config.get(boat_name, 'l'))
		self.b = float(config.get(boat_name, 'b'))
		self.m = float(config.get(boat_name, 'm'))
		self.ine_xx = float(config.get(boat_name, 'ine_xx'))
		self.ine_yy = float(config.get(boat_name, 'ine_yy'))
		self.ine_zz = float(config.get(boat_name, 'ine_zz'))
		self.x_u_ = float(config.get(boat_name, 'x_u_'))
		self.x_u = float(config.get(boat_name, 'x_u'))
		self.x_uu = float(config.get(boat_name, 'x_uu'))
		self.x_vv = float(config.get(boat_name, 'x_vv'))
		self.x_rr = float(config.get(boat_name, 'x_rr'))
		self.y_r_ = float(config.get(boat_name, 'y_r_'))
		self.y_v_ = float(config.get(boat_name, 'y_v_'))
		self.y_v = float(config.get(boat_name, 'y_v'))
		self.y_vv = float(config.get(boat_name, 'y_vv'))
		self.y_rr = float(config.get(boat_name, 'y_rr'))
		self.y_vr = float(config.get(boat_name, 'y_vr'))
		self.y_d_r = float(config.get(boat_name, 'y_d_r'))
		self.z_q_ = float(config.get(boat_name, 'z_q_'))
		self.z_w_ = float(config.get(boat_name, 'z_w_'))
		self.z_q = float(config.get(boat_name, 'z_q'))
		self.z_w = float(config.get(boat_name, 'z_w'))
		self.z_ww = float(config.get(boat_name, 'z_ww'))
		self.z_d_s = float(config.get(boat_name, 'z_d_s'))
		self.k_p_ = float(config.get(boat_name, 'k_p_'))
		self.k_p = float(config.get(boat_name, 'k_p'))
		self.k_pp = float(config.get(boat_name, 'k_pp'))
		self.k_vq = float(config.get(boat_name, 'k_vq'))
		self.k_wr = float(config.get(boat_name, 'k_wr'))
		self.m_q_ = float(config.get(boat_name, 'm_q_'))
		self.m_q = float(config.get(boat_name, 'm_q'))
		self.m_qq = float(config.get(boat_name, 'm_qq'))
		self.m_w_ = float(config.get(boat_name, 'm_w_'))
		self.m_d_s = float(config.get(boat_name, 'm_d_s'))
		self.n_r_ = float(config.get(boat_name, 'n_r_'))
		self.n_v_ = float(config.get(boat_name, 'n_v_'))
		self.n_v = float(config.get(boat_name, 'n_v'))
		self.n_vv = float(config.get(boat_name, 'n_vv'))
		self.n_r = float(config.get(boat_name, 'n_r'))
		self.n_rr = float(config.get(boat_name, 'n_rr'))
		self.n_d_r = float(config.get(boat_name, 'n_d_r'))
		self.K = float(config.get(boat_name, 'K'))
		self.T = float(config.get(boat_name, 'T'))
		self.alpha = float(config.get(boat_name, 'alpha'))
		self.deltam = float(config.get(boat_name, 'deltam'))

	def rov_select(self, filename, boat_name):
		config = readIni.readIni(filename)
		print(f'ini file is {filename}')
		self.L = float(config.get(boat_name, 'L'))
		self.B = float(config.get(boat_name, 'B'))
		self.H = float(config.get(boat_name, 'H'))
		self.Delta = float(config.get(boat_name, 'Delta'))
		self.m = float(config.get(boat_name, 'm'))
		self.x_g = float(config.get(boat_name, 'x_g'))
		self.y_g = float(config.get(boat_name, 'y_g'))
		self.z_g = float(config.get(boat_name, 'z_g'))
		self.x_b = float(config.get(boat_name, 'x_b'))
		self.y_b = float(config.get(boat_name, 'y_b'))
		self.z_b = float(config.get(boat_name, 'z_b'))
		self.Delta = float(config.get(boat_name, 'Delta'))
		self.rho_w = float(config.get(boat_name, 'rho_w'))
		self.ine_xx = float(config.get(boat_name, 'ine_xx'))
		self.ine_yy = float(config.get(boat_name, 'ine_xx'))
		self.ine_zz = float(config.get(boat_name, 'ine_xx'))
		self.x_u_ = float(config.get(boat_name, 'x_u_'))
		self.x_u = float(config.get(boat_name, 'x_u'))
		self.x_uu = float(config.get(boat_name, 'x_uu'))
		self.y_v_ = float(config.get(boat_name, 'y_v_'))
		self.y_vv = float(config.get(boat_name, 'y_vv'))
		self.y_rr = float(config.get(boat_name, 'y_rr'))
		self.z_w_ = float(config.get(boat_name, 'z_w_'))
		self.z_w = float(config.get(boat_name, 'z_w'))
		self.z_ww = float(config.get(boat_name, 'z_ww'))
		self.k_p_ = float(config.get(boat_name, 'k_p_'))
		self.k_p = float(config.get(boat_name, 'k_p'))
		self.k_pp = float(config.get(boat_name, 'k_pp'))
		self.m_q_ = float(config.get(boat_name, 'm_q_'))
		self.m_q = float(config.get(boat_name, 'm_q'))
		self.m_qq = float(config.get(boat_name, 'm_qq'))
		self.n_r_ = float(config.get(boat_name, 'n_r_'))
		self.n_r = float(config.get(boat_name, 'n_r'))
		self.n_rr = float(config.get(boat_name, 'n_rr'))