#风扰动
import numpy as np
import math
from algo_struct import algo
import configparser
def readIni(filename):
	config = configparser.ConfigParser()
	config.read(filename, encoding='utf-8')
	return config

class myWind:
	def __init__(self):
		self.CX = 0.0  # 风压系数x
		self.CY = 0.0  # 风压系数y
		self.CK = 0.0  # 风压系数k
		self.CN = 0.0  # 风压系数n
		self.Xw = 0.0  # 风阻力
		self.Yw = 0.0
		self.Kw = 0.0  # 风阻力矩
		self.Nw = 0.0
		self.Vr = 0.0  # 海平面10m处的风速
		self.h = 0.0  # 时间步长
		self.ALw = 0.0  # 横向投影面积
		self.AFw = 0.0  # 纵向投影面积
		self.HLw = 0.0  #
		self.Loa = 0.0  # 船长
		self.rho = 0.0  # 空气密度
		self.beta_w = 0.0  #绝对风向角
		self.gamma_e = 0.0
		self.sH = 0.0  # sH = HLw 横向投影中心到水线面中心垂向距离
		self.sL = 0.0  # 横向投影中心到水线面纵向距离
		self.vessel_no = 0  # 船的类型
		self.windLev = 0  # 风等级
		self.CDt = 0.0
		self.CDl = 0.0
		self.CDlAF = 0.0
		self.B = 0.0
		self.delta = 0.0
		self.k = 0.0
		self.HM = 0.0

class myDynWind:
	def __init__(self):
		self.u = 0.0
		self.v = 0.0
		self.theta = 0.0
		self.windList = ['clam','lightAir','lightBreeze',
		                 'gentleBreeze','moderateBreeze',
		                 'freshBreeze','strongBreeze',
		                 'moderateGale','freshGale',
		                 'strongGale','wholeGale','Strom',
		                 'Hurricane']
		self.boatList = ['carCarrier','cargoVesselLoaded',
		                 'cargoVesselContainer','containerShip',
		                 'drillingVessel','feery','fishingVessel',
		                 'gasTanker','offshoreSupplyVessel',
		                 'passengerLiner','researchVessel',
		                 'speedBoat','tankerLoaded','TankerBallast',
		                 'Tender']
		self.tempList = ['-10','-5','0','5','10','15','20','25','30']
		self.rhoWind = [1.342,1.317,1.292,1.269,1.247,1.225,
		                1.204 , 1.184 , 1.165]
		self.windLevel = [1/2.0, 5/2.0, (4+7)/2.0, (8+11)/2.0,
		                  (12 + 16) / 2 , (17 + 21) / 2 ,
		                  (22 + 27) / 2 , (28 + 33) / 2,
		                  (34 + 40) / 2 , (41 + 48) / 2 ,
		                  (49 + 56) / 2 , (57 + 65) / 2 ,
		                  65]
		self.windLow = [ 0 , 2 , 4 , 8 , 12 , 17 , 22 , 28 ,
		                 34 , 41 , 49 , 57]
		self.windTop = [ 1 , 3 , 7 , 11 , 16 , 21 , 27 , 33 ,
		                 40 , 48 , 56 , 65]
		self.someWind = myWind()

	def boat_select(self, filename, boat_name):
		# 新尺度的船只
		config = readIni(filename)

		self.someWind.AFw = float(config.get(boat_name,'AFw'))
		self.someWind.ALw = float(config.get(boat_name, 'ALw'))
		self.someWind.Loa = float(config.get(boat_name, 'l'))
		self.someWind.B = float(config.get(boat_name, 'B'))
		self.someWind.sL = float(config.get(boat_name, 'sL'))
		self.someWind.sH = float(config.get(boat_name, 'sH'))
		self.someWind.HLw = self.someWind.sH
		self.someWind.HM = self.someWind.ALw / self.someWind.Loa

		self.someWind.vessel_no = int(config.get(boat_name, 'vessel_no'))#船型号
		self.someWind.windLev = int(config.get(boat_name, 'windLevel'))#根据风级选择风速
		self.someWind.Vr = 0.514*self.windLevel[self.someWind.windLev]# 风速
		self.someWind.rho = self.rhoWind[int(config.get(boat_name, 'temp'))] #空气密度
		self.someWind.beta_w = float(config.get(boat_name, 'beta_w')) * math.pi/180.0#绝对风向

	def blendermann(self, boatV, psi):
		#boatV 船速度
		#psi 船航向角
		u = boatV[0]
		v = boatV[1]
		uw = self.someWind.Vr * np.cos(self.someWind.beta_w - psi)
		vw = self.someWind.Vr * np.sin(self.someWind.beta_w - psi)
		urw = u - uw
		vrw = v - vw
		Vrw = np.sqrt(urw**2 + vrw**2)
		gama_rw = -math.atan2(vrw, urw)
		#相对风向角，与船静态的时候，gama_rw结果应该一致，这个量一样其他
		algo.rad_limit(gama_rw)
		CDt =[ 0.95 , 0.85 , 0.85 , 0.85 , 0.9 , 0.85 , 0.9,
		       1 , 0.9 , 0.95, 0.7 , 0.9 , 0.9 , 0.85 , 0.9,
		       0.7 , 0.7 , 0.85]
		if(abs(gama_rw) <= np.pi/2):
			CDlAF = [0.55 , 0.65 , 0.55 , 0.55 , 0.6 , 0.6 ,
			         0.85 , 0.45 , 0.7, 0.6 , 0.55 , 0.4 ,
			         0.55 , 0.55 , 0.9 , 0.75 , 0.55]
		else:
			CDlAF = [ 0.6 , 0.55 , 0.5 , 0.55 , 0.65 , 0.8 ,
			          0.925 , 0.5 , 0.7 , 0.65 , 0.8 , 0.4 ,
			          0.65 , 0.6 , 0.55 , 0.55 , 0.55]
		windDelta = [0.8 , 0.4 , 0.4 , 0.4 , 0.65 , 0.55 , 0.1 ,
		             0.8 , 0.4 , 0.5 , 0.55 , 0.8 , 0.6 , 0.6 ,
		             0.4 , 0.4 , 0.65]
		windK = [1.2 , 1.7 , 1.4 , 1.4 , 1.1 , 1.7 , 1.7 , 1.1 ,
		         1.1 , 1.1 , 1.2 , 1.2 , 1.4 , 1.1 , 3.1 , 2.2 , 1.1]
		self.someWind.CDlAF = CDlAF[self.someWind.vessel_no]
		self.someWind.CDl = self.someWind.CDlAF*\
		                    (self.someWind.AFw/self.someWind.ALw)
		self.someWind.delta = windDelta[self.someWind.vessel_no]
		self.someWind.CDt = CDt[self.someWind.vessel_no]
		self.someWind.k = windK[self.someWind.vessel_no]

		self.someWind.CX = -self.someWind.CDlAF*\
		                   np.cos(gama_rw)/(1-0.5*self.someWind.delta*(1-self.someWind.CDl/self.someWind.CDt)*(math.sin(2*gama_rw)**2))
		self.someWind.CY = self.someWind.CDt * math.sin(gama_rw)/(1-0.5*self.someWind.delta*(1-self.someWind.CDl/self.someWind.CDt)
		                                                          *(math.sin(2*gama_rw)**2))
		self.someWind.CK = (self.someWind.sH / self.someWind.HM)*self.someWind.k*self.someWind.CY
		self.someWind.CN = self.someWind.CY * (self.someWind.sL/self.someWind.Loa - 0.18*(gama_rw - math.pi/2))

		qCoe = 0.5*self.someWind.rho*(Vrw**2)
		Xw = qCoe*self.someWind.CX*self.someWind.AFw
		Yw = qCoe*self.someWind.CY*self.someWind.ALw
		Kw = qCoe*self.someWind.CK*self.someWind.ALw*self.someWind.HLw
		Nw = qCoe*self.someWind.CN*self.someWind.ALw*self.someWind.Loa

		out = np.array([Xw, Yw, 0.0, Kw, 0.0, Nw])

		return out

