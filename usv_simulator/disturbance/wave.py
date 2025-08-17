from algo_struct.algo import rad_limit
import numpy as np
import configparser

def readIni(filename):
	config = configparser.ConfigParser()
	config.read(filename, encoding='utf-8')
	return config

class MyDynWave:
	def __init__(self):
		self.GMt = 0.3 
		self.GMl = 0.5 
		self.wind_u = 0  # 19.4m above waterline wind speed
		self.B = 0.6  # width
		self.d = 0.2  # T
		self.C = 0.4  # Square coefficient
		self.chi = 45 * np.pi / 180  # initial wave direction
		self.w = []  # frequency
		self.beta = []  # counter wave angle
		self.b = 0.0  # counter wave frequency
		self.T0 = 0.0
		self.k = 0.0
		self.a = 0.0
		self.deltaw = 0.05
		self.delta_beta = 8 * np.pi / 180
		self.rho_w = 1000
		self.g_e = 9.81
		self.l = 2.38
		self.S = 0.0
		self.wave_Hs = 0.0

	def boat_select(self, filename, boat_name):
		config = readIni(filename)

		self.rho_w = float(config.get(boat_name, 'rho_w'))
		self.chi = float(config.get(boat_name, 'chi'))# main wave direction
		self.l = float(config.get(boat_name, 'l')) # lenght
		self.m = float(config.get(boat_name, 'm'))# kg mass
		self.GMt = float(config.get(boat_name,'GMt'))
		self.GMl = float(config.get(boat_name, 'GMl'))
		self.wind_u = float(config.get(boat_name,'wind_u'))
		self.B = float(config.get(boat_name,'B'))
		self.d = float(config.get(boat_name,'d'))

	def pmSpectra(self, state, w):
		S = 0.0
		Hs = 2.06 * (self.wind_u ** 2) / (self.g_e ** 2)  # mean wave height
		self.wave_Hs = Hs
		if state == 1:
			# M Pm
			Tz = 0.710 * self.T0
			A = 4 * (np.pi ** 3) * (Hs ** 2) / (Tz ** 4)
			B = 16 * (np.pi ** 3) / (Tz ** 4)
			S = A * (w ** -5) * np.exp(-B * (w ** -4))
		elif state == 2:
			# JONSWAP
			T1 = 0.834 * self.T0
			gama = 3.3
			if w <= (5.24 / T1):
				delta = 0.07
			else:
				delta = 0.09
			Y = np.exp(-(((0.191 * w * T1 - 1) / (np.sqrt(2) * delta)) ** 2))
			S = 155 * ((Hs ** 2) / (T1 ** 4)) * (w ** -5) * np.exp((w ** -4) * (-944 / (T1 ** 4))) * (gama ** Y)
		else:
			print("out of pm range")
		return S

	def short_crested(self, S, w, beta, t, x, epsilon, mode=2):
		zeta_ij = 0.0
		if -0.5 * np.pi < beta < 0.5 * np.pi:
			fbeta = ((np.cos(beta)) ** 2) * 2 / np.pi
		else:
			fbeta = 0
		S = S * fbeta
		if mode == 1:
			zeta_ij = np.sqrt(2 * S * self.deltaw * self.delta_beta) * np.cos(
				self.k * x(1) * np.sin(beta) + self.k * x(2) * np.cos(beta) - w * t + epsilon)
		elif mode == 2:
			zeta_ij = np.sqrt(2 * S * self.deltaw * self.delta_beta)

		return zeta_ij


	def long_crested(self, w, t, chi, x, epsilon, mode=2):
		zeta_i = 0.0
		if mode == 1:

			Ak = np.sqrt(2 * self.S * self.deltaw)
			zeta_i = Ak * np.cos(w * t + epsilon - self.k * (x(1) * np.cos(chi) - x(2) * np.sin(chi)))
		elif mode == 2:
			zeta_i = np.sqrt(2 * self.S * self.deltaw)
		return zeta_i


	def tau_first_force_i(self, beta, u, t, epsilon, zeta):
		E = self.k * np.sin(beta) / 2
		F = self.k * np.cos(beta) / 2
		l = np.sqrt(self.C) * self.l
		q = u[4]
		BETA_MAT = np.zeros(6, dtype=float)
		BETA_MAT[0] = (2 * self.d / E) * np.sin(F * l) * np.sin(self.b * t + epsilon)
		BETA_MAT[1] = -2 * self.d * l / (F * self.B) * np.sin(F * l) * np.sin(self.b * t + epsilon)
		BETA_MAT[2] = (self.k * self.d / E) * np.sin(F * l) * np.cos(self.b * t + epsilon)
		BETA_MAT[3] = (self.d ** 2) / (F * self.B) * np.sin(F * l) * np.sin(self.b * t + epsilon)
		BETA_MAT[4] = (self.d / (2 * E * (F ** 2))) * (np.sin(F * l) - F * l * np.cos(F * l)) * np.sin(self.b * t + epsilon)
		BETA_MAT[5] = (self.d / (self.k * (F ** 2))) * (np.sin(F * l) - F * l * np.cos(F * l)) * np.cos(
			self.b * t + epsilon)

		# beta_1 = self.rho_w*self.g_e*zeta*np.exp(-k*d)*np.sin(E*B)*BETA_MAT
		beta_1 = self.rho_w * self.g_e * zeta * np.exp(-self.k * self.d) * E * self.B * BETA_MAT
		return beta_1


	def tau_first_force_i_j(self, beta, w, zeta):
		lamb = 2 * np.pi * w / self.g_e
		c = np.zeros(3)
		c[0] = 0.05 - 0.2 * lamb / self.l + 0.75 * ((lamb / self.l) ** 2) - 0.51 * ((lamb / self.l) ** 3)
		c[1] = 0.46 + 0.683 * lamb / self.l - 15.65 * ((lamb / self.l) ** 2) + 8.44 * ((lamb / self.l) ** 3)
		c[2] = -0.11 + 0.68 * lamb / self.l - 0.79 * ((lamb / self.l) ** 2) + 0.21 * ((lamb / self.l) ** 3)

		BETA_MAT = np.zeros(6, dtype=float)
		BETA_MAT[0] = self.l * c[0] * np.cos(beta)
		BETA_MAT[1] = self.l * c[1] * np.sin(beta)
		BETA_MAT[5] = (self.l ** 2) * c[2] * np.sin(beta)

		beta_2 = self.rho_w * self.g_e * (zeta ** 2) / 2 * BETA_MAT
		return beta_2

	def wave_param_init(self, control):
		w0 = 0.4  # set w0 or T0
		self.T0 = 2 * np.pi / w0
		self.a = self.chi - control.x[5]
		self.b = abs(w0 - ((w0 ** 2) / self.g_e) * (
				control.u[0] * np.cos(self.a) + control.u[1] * np.sin(self.a)))
		# sample frequency
		wmin = 0.2
		wmax = 1.5
		N = round((wmax - wmin) / self.deltaw)  # N in [50, 100]
		self.w = np.linspace(wmin, wmax, N)

		M = 8  # beta
		self.beta = []
		for i in range(M):
			self.beta.append(rad_limit(self.chi + i * self.delta_beta))
			self.beta.append(rad_limit(self.chi - i * self.delta_beta))

	def wave_force_cal(self, t, control):
		tau_1_ij = np.zeros(6, dtype=float)
		tau_2_ij = np.zeros(6, dtype=float)
		for i in range(len(self.w)):
			self.k = np.sqrt((self.w[i] ** 2) / self.g_e)
			self.S = self.pmSpectra(state=1, w=self.w[i])
			self.w[i] = self.w[i] - ((self.w[i] ** 2) / self.g_e) \
			                 * np.sqrt(control.u[0] ** 2 + control.u[1] ** 2) * np.cos(self.a)
			for j in range(len(self.beta)):
				# random phase
				epsilon = np.random.rand(1) * 2 * np.pi
				zeta_ij = self.short_crested(S=self.S,
								w=self.w[i],
								beta=self.beta[j],
								t=t,
								x=control.x,
								epsilon=epsilon)
				
				beta_1_ij = self.tau_first_force_i(beta=self.beta[j],
									u=control.u,
									t=t,
									epsilon=epsilon,
									zeta=zeta_ij)

				beta_2_ij = self.tau_first_force_i_j(beta=self.beta[j],
									w=self.w[i],
									zeta=zeta_ij)
				
				tau_1_ij += beta_1_ij
				tau_2_ij += beta_2_ij
		return tau_1_ij + tau_2_ij