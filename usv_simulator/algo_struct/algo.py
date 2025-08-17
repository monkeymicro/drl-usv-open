from cmath import inf, nan
from numpy import pi, floor, sqrt, sin, cos
from math import acos, asin
import numpy as np
from scipy.interpolate import pchip_interpolate, CubicSpline
import csv

def trans(point=np.zeros(3, dtype=float), angle=np.zeros(3, dtype=float), tran=np.zeros(3, dtype=float), mode=True):
	psi = angle[0]  # x
	phi = angle[1]  # y
	theta = angle[2]  # z
	# print('theta', theta)
	point = np.array([point[0], point[1], point[2], 1])

	Rx = np.array([[1, 0, 0, 0],
				   [0, cos(psi), sin(psi), 0],
				   [0, -sin(psi), cos(psi), 0],
				   [0, 0, 0, 1]])
	# print(Rx)
	Ry = np.array([[cos(phi), 0, -sin(phi), 0],
				   [0, 1, 0, 0],
				   [sin(phi), 0, cos(phi), 0],
				   [0, 0, 0, 1]])
	Rz = np.array([[cos(theta), sin(theta), 0, 0],
				   [-sin(theta), cos(theta), 0, 0],
				   [0, 0, 1, 0],
				   [0, 0, 0, 1]])
	t = np.array([[1, 0, 0, tran[0]],
				  [0, 1, 0, tran[1]],
				  [0, 0, 1, tran[2]],
				  [0, 0, 0, 1]])
	# print('Rz', Rz)
	Rxt = np.array([[1, 0, 0, 0],
				   [0, cos(psi), -sin(psi), 0],
				   [0, sin(psi), cos(psi), 0],
				   [0, 0, 0, 1]])
	# print(Rxt)
	Ryt = np.array([[cos(phi), 0, sin(phi), 0],
				   [0, 1, 0, 0],
				   [-sin(phi), 0, cos(phi), 0],
				   [0, 0, 0, 1]])
	Rzt = np.array([[cos(theta), -sin(theta), 0, 0],
				   [sin(theta), cos(theta), 0, 0],
				   [0, 0, 1, 0],
				   [0, 0, 0, 1]])
	if mode:
		T = np.dot(np.dot(np.dot(Rx, Ry), Rz), t)
	else:
		T = np.dot(np.dot(np.dot(Rxt, Ryt), Rzt), t)


	return np.dot(T, point)[0:3]

def rad_limit(rad: float) -> float:
	rad = rad - 2 * pi * floor(rad / (2 * pi))
	if rad >= pi:
		rad = rad - 2 * pi
	if rad < -pi:
		rad = rad + 2 * pi
	return rad


def get_cross_point(a=np.zeros(3), b=np.zeros(3)):
	point = np.zeros(3)
	flag = False
	try:
		if a[0] * b[1] == a[1] * b[0]:
			raise OSError('parallel lines')
		else:
			x = (b[2] * a[1] - a[2] * b[1]) / (a[0] * b[1] - b[0] * a[1])
			y = (a[2] * b[0] - b[2] * a[0]) / (a[0] * b[1] - b[0] * a[1])
			point[0] = x
			point[1] = y
	except OSError:
		return point, flag
	flag = True
	return point, flag


def get_h_line(point, n):
	eta = 1e-5
	coe = [0, 0, 0]
	if abs(vec_dot(n, np.array([0, 1, 0]))) < eta:
		coe[0] = 1
		coe[1] = 0
		coe[2] = point[0]
	else:
		coe[0] = n[1] / n[0]
		coe[1] = -1
		coe[2] = point[1] - coe[0] * point[0]
	# print('coe', coe)
	return coe


def vector_dot_angle(A, B) -> float:
	val = (A[0] * B[0] + A[1] * B[1]) / (sqrt((A[0] ** 2) + (A[1] ** 2)) * sqrt((B[0] ** 2) + (B[1] ** 2)))
	if val == inf:
		val = 0
	elif val ==nan:
		val = 0

	angle = acos(round(val, 5))
	C = vec_cross(A, B)
	flag = 0
	if C[2] > 0:
		flag = 1
	else:
		flag = -1
	return flag * angle


def vec_dot(a=np.zeros(3), b=np.zeros(3)):
	c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
	return c


def vec_cross(a=np.zeros(3, dtype=float), b=np.zeros(3, dtype=float)) -> list:
	'''
	矢量叉乘 从a向b
	:param a:
	:param b:
	:return:
	'''
	c = np.zeros(3)
	c[0] = a[1] * b[2] - a[2] * b[1]
	c[1] = -a[0] * b[2] + b[0] * a[2]
	c[2] = a[0] * b[1] - a[1] * b[0]
	return c


def my_interpolate(x_observed, y_observed, method='pchip', p_num=50):
	x, y = [], []
	if method == 'pchip':
		x = np.linspace(min(x_observed), max(x_observed), num=p_num)
		y = pchip_interpolate(x_observed, y_observed, x)
	elif method == 'cubic':
		x = np.linspace(min(x_observed), max(x_observed), num=p_num)
		cs = CubicSpline(x_observed, y_observed)
		y = cs(x)
	return x, y

def norm(a):
	result = 0
	for i in a:
		result += i ** 2
	return np.sqrt(result)

def get_circle(point, r):
	ta = np.linspace(0, 2 * pi, 360)
	x_c = point[0] + r * cos(ta)
	y_c = point[1] + r * sin(ta)
	return x_c, y_c

def in_or_out_poly(point, poly):
	flag = False
	for i in range(len(poly)):
		if i<len(poly)-1:
			a_p = poly[i]
			b_p = poly[i+1]
		else:
			a_p = poly[i]
			b_p = poly[0]
		vec_a = a_p - point
		vec_b = b_p - point
		toward = vec_cross(vec_a, vec_b)
		if toward[2] >= 0:
			flag = False
			return flag
		elif toward[2] < 0:
			flag = True

	if flag == True:
		return flag

def case_in_or_not_poly():
	square = np.array([[0,100,0],[50,100,0],[50,0,0],[0,0,0]])
	pp = np.array([200,10,0])
	flag = in_or_out_poly(pp, square)
	print(flag)


def in_half_plane(hp, point):
	flag = False
	A = hp[0]
	B = hp[1]
	C = hp[2]
	vec_n = [hp[3], hp[4], hp[5]]
	if B == 0:
		y = 0
		x = -C / A

	else:
		x = 0
		y = -C / B
	vec = point - np.array([x, y, 0])
	angle = vector_dot_angle(vec, vec_n)
	if abs(angle) <= 0.5 * pi:
		flag = True
	else:
		flag = False

	return flag


def point_to_line(a, b):
	dis = abs(a[0] * b[0] + a[1] * b[1] + a[2]) / sqrt(a[0] ** 2 + a[1] ** 2)
	return dis
