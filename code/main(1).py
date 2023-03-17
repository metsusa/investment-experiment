#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import cos, sin, acos, asin, pi, atan, fabs, ceil, floor
from collections import defaultdict
import pulp as pl
import random
import time
from matplotlib import colors, cm
from math import ceil
import csv
import copy as cp


# GRACCF的约束条件
class Assignment:
	@classmethod
	# maxDrone: the amount of the drones
	def KM(cls, Q, La, L, dimension_relationMat=[]):
		row = len(Q)
		col = len(Q[0])
		len_relationMat = len(dimension_relationMat)

		# build a optimal problem
		pro = pl.LpProblem('Max(connection and coverage)', pl.LpMaximize)
		# build variables for the optimal problem
		lpvars = [[pl.LpVariable("x"+str(i)+"y"+str(j), lowBound = 0, upBound = 1, cat='Integer') for j in range(col)] for i in range(row)]

		lpvars_GRACCF = [pl.LpVariable("val_k"+str(k), lowBound = 0, upBound = 1, cat='Integer') for k in range(len_relationMat)]

		# build optimal function
		all = pl.LpAffineExpression()
		for i in range(0,row):
			for j in range(0,col):
				all += Q[i][j]*lpvars[i][j]

		# 目标函数加入GRACCF部分
		for k in range(0, len_relationMat):
			all += dimension_relationMat[k][4]*Q[dimension_relationMat[k][0]][dimension_relationMat[k][1]]*lpvars_GRACCF[k]

		pro += all
		# build constraint for each role
		for j in range(0,col):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j],1) for i in range(0,row)]) , 0,"L"+str(j),L[j])

		# build constraint for each agent
		for i in range(0,row):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j],1) for j in range(0,col)]) , -1,"La"+str(i), La[i])

		# # 引入GRACCF中的约束15-16
		# for k in range(0, len_relationMat):
		# 	pro += pl.LpConstraint(pl.LpAffineExpression([(lpvars_GRACCF[k],2 , -1, "T'_cons15_"+str(k), lpvars[dimension_relationMat[k][0]][dimension_relationMat[k][1]]+lpvars[dimension_relationMat[k][2]][dimension_relationMat[k][3]])

		# for k in range(0, len_relationMat):
		# 	pro += pl.LpConstraint(pl.LpAffineExpression([(lpvars_GRACCF[k],1)]) , 1, "T'_cons16_"+str(k), lpvars[dimension_relationMat[k][0]][dimension_relationMat[k][1]]+lpvars[dimension_relationMat[k][2]][dimension_relationMat[k][3]]-1)
		# 引入GRACCF中的约束15-16
		for k in range(0, len_relationMat):
			pro += lpvars_GRACCF[k]*2 <= lpvars[dimension_relationMat[k][0]][dimension_relationMat[k][1]]+lpvars[dimension_relationMat[k][2]][dimension_relationMat[k][3]]

		for k in range(0, len_relationMat):
			pro += lpvars_GRACCF[k]+1 >= lpvars[dimension_relationMat[k][0]][dimension_relationMat[k][1]]+lpvars[dimension_relationMat[k][2]][dimension_relationMat[k][3]]

		# solve optimal problem
		status = pro.solve()
		print("Assignment Status: ", pl.LpStatus[status])
		print("Final Assignment Result", pl.value(pro.objective))

		# get the result of T matrix
		T = [[ lpvars[i][j].varValue for j in range(col) ] for i in range(row)]
		return [T,pl.value(pro.status),pl.value(pro.objective)]

# 计算传递闭包, type == 1 表示正类， type == -1 表示负类
def calTransitiveClosure(relationMat, typ = 1):
	resMat = cp.deepcopy(relationMat)
	MatLength = len(relationMat)
	for k in range(MatLength): # 经过中间节点k中转，能更新多少个传递关系
		for i in range(MatLength):
			for j in range(MatLength):
				if resMat[i][j] != 0:continue
				resMat[i][j] = typ * resMat[i][k]*resMat[k][j]
	return resMat

# 计算欧性闭包, type == 1 表示正类， type == -1 表示负类
def calEuclideanClosure(relationMat, typ = 1):
	resMat = cp.deepcopy(relationMat)
	MatLength = len(relationMat)
	for k in range(MatLength): # 经过中间节点k中转，能更新多少个欧性关系
		for i in range(MatLength):
			for j in range(MatLength):
				if resMat[i][j] != 0 or i == j:continue
				resMat[i][j] = typ * resMat[k][i]*resMat[k][j]
	return resMat


# 生成随机的ACC矩阵
def genRandRelationMat(matLength, times = 3): # times表示ACC矩阵的冲突数
	resMat = np.zeros((matLength, matLength))
	while times:
		i = random.randint(0, matLength-1)
		j = random.randint(0, matLength-1)
		if i != j and resMat[i][j] == 0:
			resMat[i][j] = -1 + 2*np.random.random()
			times -= 1
	return resMat

# 获取处理后的ACC矩阵 (type = 1 正关系 type = -1 负关系)
def genRelationMat(relationMat, type = 1): # times表示ACC矩阵的冲突数
	resMat = cp.deepcopy(relationMat)
	if type == 1:
		resMat[resMat < 0] = 0
	else:
		resMat[resMat > 0] = 0
	return resMat


# KD45算法对ACC矩阵进行拓展
def KD45_Algorithm(relationMat):
	positive_relationMat = genRelationMat(relationMat, 1)
	negative_relationMat = genRelationMat(relationMat, -1)
	positive_relationMat = calTransitiveClosure(positive_relationMat, 1)
	negative_relationMat = calTransitiveClosure(negative_relationMat, -1)
	positive_relationMat = calEuclideanClosure(positive_relationMat, 1)
	negative_relationMat = calEuclideanClosure(negative_relationMat, -1)
	resMat = positive_relationMat+negative_relationMat
	resMat [relationMat != 0] = 0
	return resMat+relationMat

# 利用tao值去更新ACC矩阵, type = 1 高合作，低冲突（理想团队）， type = 0 考虑合作的情况下，筛选低冲突， type = 1 考虑冲突的情况下，筛选合作
def genRelationMat_withTao(relationMat, tao_positive, type = 1):
	positive_relationMat = genRelationMat(relationMat, 1)
	negative_relationMat = genRelationMat(relationMat, -1)
	if type == 0:
		positive_relationMat[positive_relationMat < 0] = 0
		negative_relationMat[negative_relationMat > tao_positive-1] = 0
	elif type == 1:
		positive_relationMat[positive_relationMat < tao_positive] = 0
		negative_relationMat[negative_relationMat > tao_positive-1] = 0
	else:
		positive_relationMat[positive_relationMat < tao_positive] = 0
		negative_relationMat[negative_relationMat > 0] = 0
	return positive_relationMat+negative_relationMat;


# 生成随机的Q矩阵
def genRandQMat(agentNum, roleNum):
	resMat = np.zeros((agentNum, roleNum))
	for i in range(agentNum):
		for j in range(roleNum):
			resMat[i][j] = round(np.random.random(), 2)
	return resMat

# 对关系矩阵降维处理，输出
def dimensionalityReduction(relationMat, correspondingList):
	resMat = []
	for i in range(len(relationMat)):
		for j in range(len(relationMat[0])):
			if relationMat[i][j] != 0:
				resMat.append([int(correspondingList[i][0]), int(correspondingList[i][1]), int(correspondingList[j][0]), int(correspondingList[j][1]), relationMat[i][j]])
	return resMat

# 复现GRACCF中的关系矩阵
def recurGRACCF_Mat(matLength):
	resMat = np.zeros((matLength, matLength))
	resMat[1][5]=-0.3;resMat[1][6]=0.35;resMat[1][7]=0.35;resMat[1][20]=-0.4;resMat[1][50]=0.8;resMat[1][51]=0.9;

	resMat[2][5]=-0.2;resMat[2][6]=-0.2;resMat[2][7]=0.2;resMat[2][20]=-0.5;resMat[2][50]=0.5;resMat[2][51]=0.6;

	resMat[5][1]=-0.2;resMat[5][2]=0.2;resMat[5][17]=0.2;resMat[5][18]=0.2;resMat[5][19]=0.3;resMat[5][44]=-0.3;resMat[5][46]=0.35;resMat[5][50]=0.7;resMat[5][51]=0.6;

	resMat[6][1]=-0.35;resMat[6][2]=-0.2;resMat[6][17]=0.2;resMat[6][18]=0.2;resMat[6][19]=0.3;resMat[6][44]=-0.2;resMat[6][46]=-0.2;

	resMat[7][1]=0.35;resMat[7][2]=0.4;resMat[7][17]=0.2;resMat[7][18]=0.2;resMat[7][19]=0.2;resMat[7][44]=-0.3;resMat[6][46]=0.35;

	resMat[17][5]=0.2;resMat[17][6]=0.2;resMat[17][7]=0.3;resMat[17][20]=-0.5;resMat[17][21]=-0.4;resMat[17][23]=-0.3;resMat[17][50]=0.6;resMat[17][51]=0.7;

	resMat[18][5]=0.2;resMat[18][6]=0.2;resMat[18][7]=0.3;resMat[18][20]=-0.4;resMat[18][21]=-0.45;resMat[18][23]=-0.3;

	resMat[19][5]=0.2;resMat[19][6]=0.2;resMat[19][7]=0.2;resMat[19][20]=-0.2;resMat[19][21]=-0.2;resMat[19][23]=-0.3;

	resMat[20][17]=0.3;resMat[20][18]=0.2;resMat[20][19]=0.3;resMat[20][50]=0.8;resMat[20][51]=0.7;


	resMat[21][17]=-0.4;resMat[21][18]=-0.45;resMat[21][19]=-0.2;resMat[21][50]=0.6;resMat[21][51]=0.5;

	resMat[23][17]=-0.2;resMat[23][18]=-0.2;resMat[23][19]=-0.3;

	resMat[44][1]=0.3;resMat[44][2]=0.2;resMat[44][5]=-0.3;resMat[44][6]=0.35;resMat[44][7]=0.35;resMat[44][17]=0.3;resMat[44][18]=0.2;resMat[44][19]=0.1;resMat[44][20]=-0.5;resMat[44][21]=-0.4;
	resMat[44][23]=-0.3;resMat[44][50]=0.7;resMat[44][51]=0.7;

	resMat[46][5]=-0.2;resMat[46][6]=-0.2;resMat[46][7]=0.2;resMat[46][50]=0.6;resMat[46][51]=0.6;

	resMat[50][20]=-0.4;resMat[50][44]=0.8;resMat[50][46]=0.6;

	resMat[51][20]=-0.3;resMat[51][44]=0.8;resMat[51][46]=0.6;

	return resMat




if __name__ == '__main__':
	positionType = 4
	empolyeeNum = 13
	matLength = positionType * empolyeeNum
	L = [1, 2, 4, 2]
	La = np.ones(empolyeeNum)
	Q_Matrix = [
		[0.18, 0.82, 0.29, 0.01],
		[0.35, 0.80, 0.58, 0.35],
		[0.84, 0.85, 0.86, 0.36],
		[0.96, 0.51, 0.45, 0.64],
		[0.22, 0.33, 0.68, 0.33],
		[0.96, 0.50, 0.10, 0.73],
		[0.25, 0.18, 0.23, 0.39],
		[0.56, 0.35, 0.80, 0.62],
		[0.49, 0.09, 0.33, 0.58],
		[0.38, 0.54, 0.72, 0.20],
		[0.91, 0.31, 0.34, 0.15],
		[0.85, 0.34, 0.43, 0.18],
		[0.44, 0.06, 0.66, 0.37]
		]

	correspondingList = []
	for i in range(empolyeeNum):
		for j in range(positionType):
			correspondingList.append((i,j))
	# print(correspondingList)
	# relationMat = genRandRelationMat(matLength)

	# #利用KM算法求解
	# TMatrix, result, performance = Assignment.KM(Q_Matrix, La, L, dimension_relationMat)
	# # print(performance)

	# #利用GRA_CCF算法求解
	# TMatrix_kd45, result_kd45, performance_kd45 = Assignment.KM(Q_Matrix, La, L, dimension_relationMat_res)
	# # print(performance_kd45)

	#引入tao值，分析执行性能变化
	interval = 0.05
	tao_positive = 0
	# tao_negative = 0

	x = np.arange(0, 1, interval)
	y_orig = []
	y = []
	y_GRA = []
	while 1-tao_positive >= 0:
		print("------------begin")
		relationMat_orig = recurGRACCF_Mat(matLength)
		relationMat = cp.deepcopy(relationMat_orig)
		relationMat = genRelationMat_withTao(relationMat, tao_positive, 2)
		dimension_relationMat = dimensionalityReduction(relationMat, correspondingList)
		print(len(dimension_relationMat))
		relationMat = cp.deepcopy(relationMat_orig)
		res = KD45_Algorithm(relationMat)
		res = genRelationMat_withTao(res, tao_positive, 2)
		dimension_relationMat_res = dimensionalityReduction(res, correspondingList)
		print(len(dimension_relationMat_res))
		#利用KM算法求解
		TMatrix, result, performance = Assignment.KM(Q_Matrix, La, L, dimension_relationMat)
		# print(performance)
		y_orig.append(performance)

		#利用GRA_CCF算法求解
		TMatrix_kd45, result_kd45, performance_kd45 = Assignment.KM(Q_Matrix, La, L, dimension_relationMat_res)

		y.append(performance_kd45)

		#利用GRA算法求解
		TMatrix_GRA, result_GRA, performance_GRA = Assignment.KM(Q_Matrix, La, L)

		y_GRA.append(performance_GRA)

		# print(performance_kd45)
		tao_positive += interval

	print(x)
	print("GRACCF result: "+ str(y_orig))
	print("GRACCF + KD45 result: "+ str(y))
	print("GRA result: "+ str(y_GRA))
	# 画图
	ax = plt.subplot(111)
	ax.plot(x, y_orig,'r^-')
	ax.plot(x, y, color='#900302', marker='*', linestyle='-')
	ax.plot(x, y_GRA,'b+')
	plt.xlabel('tao')
	plt.ylabel('group performance')
	plt.legend(('GRACCF', 'KD45-GRACCF', 'GRA'), loc='upper right')
	plt.title('The Changes of the Group Performance')
	plt.show()
	# print(dimension_relationMat)
	# print(dimension_relationMat_res)
	# print(TMatrix)
	# print(TMatrix_kd45)










	# print("original ACC matrix: ")
	# print(relationMat)
	# print(genRandQMat(empolyeeNum, positionType))
	# print("GRACCF-KD45")