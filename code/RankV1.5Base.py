import pandas as pd
import numpy as np
from math import cos, sin, acos, asin, pi, atan, ceil, floor
from collections import defaultdict
import pulp as pl
from pulp import lpSum, LpVariable, LpContinuous, CPLEX_PY
import random
import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from math import ceil
import csv
import copy as cp
from gurobipy import *
import xlwt


# P矩阵范围值常量
NEGLOW = 1
NEGCANCELNUM = 0.05
NEGCANCEL = 51
NEGHIGH = 101
NORLOW = 101
NORHIGH = 200
SECLOW = 200
SECHIGH = 400
FIRSTLOW = 400
FIRSTHIGH = 600
# Q矩阵范围值常量
QRANGELOW = 400
QRANGEHIGH = 600
# 平均差偏移量
AVEDEVI = 0.25

class Assignment:
    @classmethod
    # maxDrone: the amount of the drones
    def KM(cls, Qp, Q, L, agentNum, roleNum, groupNum, duoRank):
        row = len(Qp)
        col = len(Qp[0])

        # build a optimal problem
        pro = pl.LpProblem('Max(connection and coverage)', pl.LpMaximize)

        solver = pl.getSolver('GUROBI_CMD')

        # build variables for the optimal problem
        lpvars = [[[pl.LpVariable("x" + str(i) + "y" + str(j) + "l" + str(k), lowBound=0, upBound=1, cat='Integer')
                     for k in range(groupNum)] 
                     for j in range(col)]
                  for i in range(row)]
        y_vars = [[pl.LpVariable("a" + str(a) + "b" + str(b), 0, 1, pl.LpBinary)
                    for b in range(groupNum)]
                      for a in range(len(duoRank))]

        # build optimal function (1)
        all = pl.LpAffineExpression()
        for i in range(0, row):
            for j in range(0, col):
                for k in range(0, groupNum):
                    all += Qp[i][j] * lpvars[i][j][k]

        pro += all

        # build constraint for each role Role L向量约束 (4)
        # LSum = 0
        # for j in range(col):
        #     for i in range(row):
        #         for k in range(groupNum):
        #             LSum += lpvars[i][j][k]
        #     pro += LSum == L[j]
        #     LSum = 0

        # build constraint for each agent agent全1指派 (5)
        ASum = 0
        for i in range(row):
            for j in range(col):
                for k in range(groupNum):
                    ASum += lpvars[i][j][k]
            pro += ASum == 1
            ASum = 0

        # build constraint for each team 一个团队只有五个人,其中每个角色只有一个人 (6)两种都可
        KSum = 0
        for k in range(groupNum):
            for j in range(col):
                for i in range(row):
                    KSum += lpvars[i][j][k]
                pro += KSum == 1
                KSum = 0

        # build constraint for duoRank agent 对双排玩家的约束，双排玩家不能两人扮演同一角色 (7) 不需要
        # duoSum = 0
        # for i in range(len(duoRank)):
        #     for j in range(col):
        #         for k in range(groupNum):
        #             duoSum += lpvars[duoRank[i][0]][j][k] + lpvars[duoRank[i][1]][j][k]
        #         pro += duoSum <= 1
        #         duoSum = 0

        # build constraint for duoRank agent 对双排玩家的约束，双排玩家必须在同一组 (8)
        # duoGSum = [[0 for j in range(groupNum)] for i in range(len(duoRank))]
        # for i in range(len(duoRank)):
        #     for k in range(groupNum):
        #         for j in range(roleNum):
        #             duoGSum[i][k] += lpvars[duoRank[i][0]][j][k] + lpvars[duoRank[i][1]][j][k]
        #         pro += duoGSum[i][k] <= 5 * y_vars[i][k]
        #         pro += 1.9 - duoGSum[i][k] <= 5 * (1 - y_vars[i][k])

        duoGSum = [[0 for j in range(groupNum)] for i in range(len(duoRank))]
        for i in range(len(duoRank)):
            for k in range(groupNum):
                for j in range(roleNum):
                    duoGSum[i][k] += lpvars[duoRank[i][0]][j][k] + lpvars[duoRank[i][1]][j][k]
                pro += duoGSum[i][k] == 2 * y_vars[i][k]

        # build constraint for duoRank agent 对双排玩家的约束，双排玩家不能有两组都在一个队伍中
        # duoGAll = [0 for i in range(groupNum)]
        # for i in range(groupNum):
        #     for j in range(len(duoRank)):
        #         for k in range(roleNum):
        #             duoGAll[i] += lpvars[duoRank[j][0]][k][i] + lpvars[duoRank[j][1]][k][i]
        # for i in range(groupNum):
        #     pro += duoGAll[i] <= 2
        # build constraint for duoRank agent 对双排玩家的约束，双排玩家负载均衡 (9)

        # duoGSum = [0 for j in range(groupNum)]
        # for k in range(groupNum):
        #     for i in range(len(duoRank)):
        #         for j in range(roleNum):
        #             duoGSum[k] += lpvars[duoRank[i][0]][j][k] + lpvars[duoRank[i][1]][j][k]
        # for k1 in range(groupNum):
        #     for k2 in range(groupNum):
        #         pro += duoGSum[k1] - duoGSum[k2] <= 2
        #         pro += duoGSum[k2] - duoGSum[k1] >= -2



        # # build constraint for group balance 组胜率平衡
        # allAve = 0
        # aveSum = [0 for i in range(groupNum)]
        #
        # for i in range(row):
        #     for j in range(col):
        #         for k in range(groupNum):
        #             allAve += Q[i][j] * lpvars[i][j][k]
        # allAve = allAve / groupNum
        #
        # for i in range(groupNum):
        #     for j in range(row):
        #         for k in range(roleNum):
        #             aveSum[i] += Q[j][k] * lpvars[j][k][i]
        #     aveSum[i] = aveSum[i]
        #
        # for i in range(groupNum):
        #     pro += aveSum[i] - allAve <= AVEDEVI
        #     pro += aveSum[i] - allAve >= -AVEDEVI



        status = pro.solve(solver)
        print("Assignment Status: ", pl.LpStatus[status])
        print("Final Assignment Result", pl.value(pro.objective))

        # get the result of T matrix
        T = [[[lpvars[i][j][k].varValue for k in range(groupNum)] for j in range(col)] for i in range(row)]
        return [T, pl.value(pro.status), pl.value(pro.objective)]

# 随机化Q矩阵 V1.3
def getQMat(agentNum, roleNum):
    Q = np.random.randint(QRANGELOW,QRANGEHIGH,[agentNum, roleNum])
    Q = Q.astype(float)
    for i in range(agentNum):
        for j in range(roleNum):
            Q[i, j] = Q[i, j] / 1000
    return Q

# 随机化P矩阵 V1.3
def getPMatRound(agentNum, roleNum, duoRank):
    Sum = [0 for i in range(agentNum)]
    P = np.zeros((agentNum, roleNum))
    PIndex = np.zeros((agentNum, roleNum))
    for i in range(agentNum):
        P[i] = random.sample(range(0, 1000), 5)
    for i in range(agentNum):
        for j in range(roleNum):
            Sum[i] += P[i][j]
    for i in range(agentNum):
        for j in range(roleNum):
            P[i][j] = P[i][j] / Sum[i]


    # 得到P矩阵大小顺序，获得偏好
    PSorted = P
    PREAL = np.zeros((agentNum, roleNum))
    PREALN0 = np.zeros((agentNum, roleNum))
    for i in range(agentNum):
        for j in range(roleNum):
            PREAL[i][j] = P[i][j]
    for i in range(agentNum):
        for j in range(roleNum):
            PREALN0[i][j] = P[i][j]

    for i in range(agentNum):
        PSorted[i] = np.sort(P[i])
    for i in range(agentNum):
        for j in range(roleNum):
            for k in range(roleNum):
                if PSorted[i][j] == PREAL[i][k]:
                    PIndex[i][k] = roleNum - j

    BFLAG = False
    for i in range(len(duoRank)):
        for k2 in range(roleNum):
            if PIndex[duoRank[i][0]][k2] == 5:
                for k3 in range(roleNum):
                    if PIndex[duoRank[i][1]][k3] == 5:
                        if (PREAL[duoRank[i][0]][k2] > 0.05) or (PREAL[duoRank[i][1]][k3] > 0.05):
                            if (PREAL[duoRank[i][0]][k2] < 0.10) and (PREAL[duoRank[i][1]][k3] < 0.10):
                                flag = np.random.randint(NEGCANCEL, NEGHIGH, [1])
                                flag = flag.astype(float)
                                flag = flag[0] / 1000
                                if (PREAL[duoRank[i][0]][k2] > flag) or (PREAL[duoRank[i][1]][k3] > flag):
                                    BFLAG = True
                                    break
                                else:
                                    print(duoRank[i][0], "因为都小于flag %f 清0" % (flag))
                                    print(PREAL[duoRank[i][0]][k2], PREAL[duoRank[i][1]][k3])
                                    PREAL[duoRank[i][0]][k2] = 0
                                    PREAL[duoRank[i][1]][k3] = 0
                                    BFLAG = True
                                    break
                            else:
                                BFLAG = True
                                break
                        else:
                            print(duoRank[i][0], "因为都小于0.05清0")
                            print(PREAL[duoRank[i][0]][k2], PREAL[duoRank[i][1]][k3])
                            PREAL[duoRank[i][0]][k2] = 0
                            PREAL[duoRank[i][1]][k3] = 0
                            BFLAG = True
                            break
                if BFLAG == True:
                    break

    duoRankList = []
    for i in range(len(duoRank)):
        duoRankList.append(duoRank[i][0])
        duoRankList.append(duoRank[i][1])

    # 清洗P矩阵负偏好
    for i in range(agentNum):
        if duoRankList.count(i) == 1:
            continue
        for j in range(roleNum):
            if PIndex[i][j] == 5:
                if PREAL[i, j] <= NEGCANCELNUM:
                    PREAL[i, j] = 0
                elif PREAL[i, j] < 0.10:
                    flag = np.random.randint(NEGCANCEL, NEGHIGH, [1])
                    flag = flag.astype(float)
                    flag = flag[0] / 1000
                    if PREAL[i, j] < flag:
                        PREAL[i, j] = 0

    # 判断双排是否有相同的第一偏好, 有的话重做双排玩家
    SumDuo = [0 for i in range(agentNum)]
    for i in range(len(duoRank)):
        for j in range(roleNum):
            while (PIndex[duoRank[i][0]][j] == PIndex[duoRank[i][1]][j]) and (PIndex[duoRank[i][0]][j] == 1) :
                PREAL[duoRank[i][0]] = random.sample(range(0, 1000), 5)
                PREAL[duoRank[i][1]] = random.sample(range(0, 1000), 5)
                for k in range(roleNum):
                    SumDuo[duoRank[i][0]] += PREAL[duoRank[i][0]][k]
                    SumDuo[duoRank[i][1]] += PREAL[duoRank[i][1]][k]
                for k in range(roleNum):
                    PREAL[duoRank[i][0]][k] = PREAL[duoRank[i][0]][k] / SumDuo[duoRank[i][0]]
                    PREAL[duoRank[i][1]][k] = PREAL[duoRank[i][1]][k] / SumDuo[duoRank[i][1]]
                PSortDuo = np.zeros((2, roleNum))
                PSortDuo[0] = PREAL[duoRank[i][0]]
                PSortDuo[1] = PREAL[duoRank[i][1]]
                PSortDuo[0] = np.sort(PSortDuo[0])
                PSortDuo[1] = np.sort(PSortDuo[1])
                for k1 in range(2):
                    for k2 in range(roleNum):
                        for k3 in range(roleNum):
                            if PSortDuo[k1][k2] == PREAL[duoRank[i][k1]][k3]:
                                PIndex[duoRank[i][k1]][k3] = roleNum - k2

                BFLAG =False
                for k2 in range(roleNum):
                    if PIndex[duoRank[i][0]][k2] == 5:
                        for k3 in range(roleNum):
                            if PIndex[duoRank[i][1]][k3] == 5:
                                if (PREAL[duoRank[i][0]][k2] > 0.05) or (PREAL[duoRank[i][1]][k3] > 0.05):
                                    if (PREAL[duoRank[i][0]][k2] < 0.10) and (PREAL[duoRank[i][1]][k3] < 0.10):
                                        flag = np.random.randint(NEGCANCEL, NEGHIGH, [1])
                                        flag = flag.astype(float)
                                        flag = flag[0] / 1000
                                        if (PREAL[duoRank[i][0]][k2] > flag) or (PREAL[duoRank[i][1]][k3] > flag):
                                            BFLAG = True
                                            break
                                        else:
                                            print(duoRank[i][0], "因为都小于flag %f 清0"%(flag))
                                            print(PREAL[duoRank[i][0]][k2], PREAL[duoRank[i][1]][k3])
                                            PREAL[duoRank[i][0]][k2] = 0
                                            PREAL[duoRank[i][1]][k3] = 0
                                            BFLAG = True
                                            break
                                    else:
                                        BFLAG = True
                                        break
                                else:
                                    print(duoRank[i][0], "因为都小于0.05清0")
                                    print(PREAL[duoRank[i][0]][k2], PREAL[duoRank[i][1]][k3])
                                    PREAL[duoRank[i][0]][k2] = 0
                                    PREAL[duoRank[i][1]][k3] = 0
                                    BFLAG = True
                                    break
                        if BFLAG == True:
                            break

    PALL = []
    PALL.append(P)
    PALL.append(PREALN0)
    PALL.append(PREAL)
    PALL.append(PIndex)
    return PALL

# 随机化P矩阵不均匀版本 V1.3
def getPMatRoundUnequal(agentNum, roleNum, duoRank):
    Sum = [0 for i in range(agentNum)]
    P = np.zeros((agentNum, roleNum))
    PIndex = np.zeros((agentNum, roleNum))
    for i in range(agentNum):
        P[i] = random.sample(range(0, 1000), 5)
    for i in range(agentNum):
        P[i][4] = random.randint(0, 200)
    for i in range(agentNum):
        for j in range(roleNum):
            Sum[i] += P[i][j]
    for i in range(agentNum):
        for j in range(roleNum):
            P[i][j] = P[i][j] / Sum[i]


    # 得到P矩阵大小顺序，获得偏好
    PSorted = P
    PREAL = np.zeros((agentNum, roleNum))
    PREALN0 = np.zeros((agentNum, roleNum))
    for i in range(agentNum):
        for j in range(roleNum):
            PREAL[i][j] = P[i][j]
    for i in range(agentNum):
        for j in range(roleNum):
            PREALN0[i][j] = P[i][j]

    for i in range(agentNum):
        PSorted[i] = np.sort(P[i])
    for i in range(agentNum):
        for j in range(roleNum):
            for k in range(roleNum):
                if PSorted[i][j] == PREAL[i][k]:
                    PIndex[i][k] = roleNum - j

    BFLAG = False
    for i in range(len(duoRank)):
        for k2 in range(roleNum):
            if PIndex[duoRank[i][0]][k2] == 5:
                for k3 in range(roleNum):
                    if PIndex[duoRank[i][1]][k3] == 5:
                        if (PREAL[duoRank[i][0]][k2] > 0.05) or (PREAL[duoRank[i][1]][k3] > 0.05):
                            if (PREAL[duoRank[i][0]][k2] < 0.10) and (PREAL[duoRank[i][1]][k3] < 0.10):
                                flag = np.random.randint(NEGCANCEL, NEGHIGH, [1])
                                flag = flag.astype(float)
                                flag = flag[0] / 1000
                                if (PREAL[duoRank[i][0]][k2] > flag) or (PREAL[duoRank[i][1]][k3] > flag):
                                    BFLAG = True
                                    break
                                else:
                                    print(duoRank[i][0], "因为都小于flag %f 清0" % (flag))
                                    print(PREAL[duoRank[i][0]][k2], PREAL[duoRank[i][1]][k3])
                                    PREAL[duoRank[i][0]][k2] = 0
                                    PREAL[duoRank[i][1]][k3] = 0
                                    BFLAG = True
                                    break
                            else:
                                BFLAG = True
                                break
                        else:
                            print(duoRank[i][0], "因为都小于0.05清0")
                            print(PREAL[duoRank[i][0]][k2], PREAL[duoRank[i][1]][k3])
                            PREAL[duoRank[i][0]][k2] = 0
                            PREAL[duoRank[i][1]][k3] = 0
                            BFLAG = True
                            break
                if BFLAG == True:
                    break

    duoRankList = []
    for i in range(len(duoRank)):
        duoRankList.append(duoRank[i][0])
        duoRankList.append(duoRank[i][1])

    # 清洗P矩阵负偏好
    for i in range(agentNum):
        if duoRankList.count(i) == 1:
            continue
        for j in range(roleNum):
            if PIndex[i][j] == 5:
                if PREAL[i, j] <= NEGCANCELNUM:
                    PREAL[i, j] = 0
                elif PREAL[i, j] < 0.10:
                    flag = np.random.randint(NEGCANCEL, NEGHIGH, [1])
                    flag = flag.astype(float)
                    flag = flag[0] / 1000
                    if PREAL[i, j] < flag:
                        PREAL[i, j] = 0

    # 判断双排是否有相同的第一偏好, 有的话重做双排玩家
    SumDuo = [0 for i in range(agentNum)]
    for i in range(len(duoRank)):
        for j in range(roleNum):
            while (PIndex[duoRank[i][0]][j] == PIndex[duoRank[i][1]][j]) and (PIndex[duoRank[i][0]][j] == 1) :
                PREAL[duoRank[i][0]] = random.sample(range(0, 1000), 5)
                PREAL[duoRank[i][1]] = random.sample(range(0, 1000), 5)
                for k in range(roleNum):
                    SumDuo[duoRank[i][0]] += PREAL[duoRank[i][0]][k]
                    SumDuo[duoRank[i][1]] += PREAL[duoRank[i][1]][k]
                for k in range(roleNum):
                    PREAL[duoRank[i][0]][k] = PREAL[duoRank[i][0]][k] / SumDuo[duoRank[i][0]]
                    PREAL[duoRank[i][1]][k] = PREAL[duoRank[i][1]][k] / SumDuo[duoRank[i][1]]
                PSortDuo = np.zeros((2, roleNum))
                PSortDuo[0] = PREAL[duoRank[i][0]]
                PSortDuo[1] = PREAL[duoRank[i][1]]
                PSortDuo[0] = np.sort(PSortDuo[0])
                PSortDuo[1] = np.sort(PSortDuo[1])
                for k1 in range(2):
                    for k2 in range(roleNum):
                        for k3 in range(roleNum):
                            if PSortDuo[k1][k2] == PREAL[duoRank[i][k1]][k3]:
                                PIndex[duoRank[i][k1]][k3] = roleNum - k2

                BFLAG =False
                for k2 in range(roleNum):
                    if PIndex[duoRank[i][0]][k2] == 5:
                        for k3 in range(roleNum):
                            if PIndex[duoRank[i][1]][k3] == 5:
                                if (PREAL[duoRank[i][0]][k2] > 0.05) or (PREAL[duoRank[i][1]][k3] > 0.05):
                                    if (PREAL[duoRank[i][0]][k2] < 0.10) and (PREAL[duoRank[i][1]][k3] < 0.10):
                                        flag = np.random.randint(NEGCANCEL, NEGHIGH, [1])
                                        flag = flag.astype(float)
                                        flag = flag[0] / 1000
                                        if (PREAL[duoRank[i][0]][k2] > flag) or (PREAL[duoRank[i][1]][k3] > flag):
                                            BFLAG = True
                                            break
                                        else:
                                            print(duoRank[i][0], "因为都小于flag %f 清0"%(flag))
                                            print(PREAL[duoRank[i][0]][k2], PREAL[duoRank[i][1]][k3])
                                            PREAL[duoRank[i][0]][k2] = 0
                                            PREAL[duoRank[i][1]][k3] = 0
                                            BFLAG = True
                                            break
                                    else:
                                        BFLAG = True
                                        break
                                else:
                                    print(duoRank[i][0], "因为都小于0.05清0")
                                    print(PREAL[duoRank[i][0]][k2], PREAL[duoRank[i][1]][k3])
                                    PREAL[duoRank[i][0]][k2] = 0
                                    PREAL[duoRank[i][1]][k3] = 0
                                    BFLAG = True
                                    break
                        if BFLAG == True:
                            break

    PALL = []
    PALL.append(P)
    PALL.append(PREALN0)
    PALL.append(PREAL)
    PALL.append(PIndex)
    return PALL

# 计算满意度
def getSatisfactionNum(agentNum, roleNum, groupNum, duoRank, PRealN0, PReal, PIndex, TMat, total, MD):
    P1Num = 0
    P2Num = 0
    P3Num = 0
    P5Num = 0
    PNO = 0
    for i in range(agentNum):
        for j in range(roleNum):
            for k in range(groupNum):
                if TMat[i][j][k] == 1 and PIndex[i][j] == 1:
                    P1Num += PRealN0[i][j]
                elif TMat[i][j][k] == 1 and PIndex[i][j] == 2:
                    P2Num += PRealN0[i][j]
                elif TMat[i][j][k] == 1 and PIndex[i][j] == 3:
                    P3Num += PRealN0[i][j]
                elif TMat[i][j][k] == 1 and PIndex[i][j] == 4:
                    P3Num += PRealN0[i][j]
                elif TMat[i][j][k] == 1 and PIndex[i][j] == 5 and PReal[i][j] > 0:
                    P5Num += PRealN0[i][j]
                elif TMat[i][j][k] == 1 and PIndex[i][j] == 5 and PReal[i][j] == 0:
                    PNO += PRealN0[i][j]
    print("偏好值")
    print(P1Num)
    print(P2Num)
    print(P3Num)
    print(P5Num)
    print(PNO)

    P2Num = 1.5 * P2Num
    P5Num = 12 * P5Num
    PNO = 12 * PNO
    total = total ** 0.5
    MD = 1000 * MD

    sa = 0
    sa = P1Num + 0.8 * P2Num - 0.8 * P5Num - PNO - total - MD
    return sa

#分析T矩阵  V1.3
def analTMat(agentNum, roleNum, groupNum, TMat, PIndex, duoRank):
    # 分析每个agent拿到的偏好位置
    PlayNum = agentNum
    Best = 0
    Second = 0
    Normal = 0
    Worst = 0
    TMatAnal = TMat[0]
    for i in range(1, agentNum):
        TMatAnal = np.vstack([TMatAnal, TMat[i]])
    for i in range(0, agentNum * roleNum, roleNum):
        for j in range(roleNum):
            for k in range(groupNum):
                if TMatAnal[i + j, k] == 1:
                    if PIndex[i // roleNum, j] == 1:
                        Best += 1
                    elif PIndex[i // roleNum, j] == 2:
                        Second += 1
                    elif (PIndex[i // roleNum, j] == 3) or (PIndex[i // roleNum, j] == 4):
                        Normal += 1
                    else:
                        Worst += 1
    print("During all matches, %d%% have their best position, %d%% have their second position, "
          "%d%% have their Normal position, %d%% have their Worst position."
          % (Best/PlayNum*100, Second/PlayNum*100, Normal/PlayNum*100, Worst/PlayNum*100))
    PosData = [Best/PlayNum, Second/PlayNum, Normal/PlayNum, Worst/PlayNum]

    # 分析每个组的agent数量
    gr = [0 for i in range(groupNum)]
    for i in range(agentNum):
        for j in range(roleNum):
            for k in range(groupNum):
                if TMat[i][j][k] == 1:
                    gr[k] += 1
    print(gr)

    # 分析每个组的role数量
    grole = [[] for i in range(groupNum)]
    for i in range(agentNum):
        for j in range(roleNum):
            for k in range(groupNum):
                if TMat[i][j][k] == 1:
                    grole[k].append(j)
    print("每个组role数")
    for i in range(groupNum):
        print(grole[i])

    # 分析双排的所在组
    for i in range(len(duoRank)):
        for j in range(roleNum):
            for k in range(groupNum):
                if TMat[duoRank[i][0]][j][k] == 1:
                    print("%d pos in %d " % (duoRank[i][0], k))
                if TMat[duoRank[i][1]][j][k] == 1:
                    print("%d pos in %d " % (duoRank[i][1], k))


    return  PosData

# 获取双排向量 V1.3
def getDRVec(agentNum, duoNum):
    duoRankPlayers = random.sample(range(0, agentNum), duoNum)
    duoRank = []
    for i in range(0, len(duoRankPlayers), 2):
        duoRank.append([duoRankPlayers[i], duoRankPlayers[i+1]])
    return duoRank

# 检测双排弃局 V1.3
def detectDuoFail(roleNum, groupNum, duoRank, PIndex, PReal, TMat):
    delRowDuo = []
    breakFlag = 0
    for i in range(len(duoRank)):
        for j in range(roleNum):
            for k in range(groupNum):
                # 检测双排的agent有没有被指派到负偏好且为0不会选择扮演的情况，有的话将双排两人添加到待删除列表
                if (TMat[duoRank[i][0]][j][k] == 1) and (PIndex[duoRank[i][0]][j] == 5) and (PReal[duoRank[i][0]][j] == 0):
                    delRowDuo.append([duoRank[i][0], duoRank[i][1]])
                    breakFlag = 1
                    break
                elif (TMat[duoRank[i][1]][j][k] == 1) and (PIndex[duoRank[i][1]][j] == 5) and (PReal[duoRank[i][1]][j] == 0):
                    delRowDuo.append([duoRank[i][0], duoRank[i][1]])
                    breakFlag = 1
                    break
            if breakFlag == 1:
                breakFlag = 0
                break
    return delRowDuo

# 检测单人弃局 要跳过双人 把duoRank化为单列表 V1.3
def detectSigFail(agentNum, roleNum, groupNum, duoRank, PIndex, PReal, TMat):
    duoRankList = []
    delRowSig = []
    for i in range(len(duoRank)):
        duoRankList.append(duoRank[i][0])
        duoRankList.append(duoRank[i][1])
    for i in range(agentNum):
        for j in range(roleNum):
            for k in range(groupNum):
                if duoRankList.count(i) == 1:
                    continue
                if (TMat[i][j][k] == 1) and (PIndex[i][j] == 5) and (PReal[i][j] == 0):
                    delRowSig.append(i)
    return delRowSig

# Q归一 V1.3
def getNormalizedQ(Q):
    max = min = Q[0][0]
    row = len(Q)
    column = len(Q[0])
    for i in range(row):
        for j in range(column):
            if (Q[i, j] > max):
                max = Q[i, j]
            if (Q[i, j] < min):
                min = Q[i, j]
    for i in range(row):
        for j in range(column):
            Q[i, j] = round((Q[i, j] - min) / (max - min), 2)
    return Q

# 计算组的平均差 V1.3
def calGroupMD(Q, agentNum, roleNum, groupNum, TMat):
    allAve = 0
    groupAve = []
    aveSum = 0
    MD = 0

    for i in range(agentNum):
        for j in range(roleNum):
            for k in range(groupNum):
                if TMat[i][j][k] == 1:
                    allAve += Q[i][j]
    allAve = allAve / groupNum

    for k in range(groupNum):
        for i in range(agentNum):
            for j in range(roleNum):
                if TMat[i][j][k] == 1:
                    aveSum += Q[i][j]
        groupAve.append(aveSum)
        aveSum = 0

    for i in range(groupNum):
        MD += abs(groupAve[i] - allAve)
    MD = MD / groupNum
    print(allAve)
    print(groupAve)
    print(MD)

    print(np.std(groupAve,ddof=1))
    print(np.var(groupAve))

    return MD

# 计算双排负偏好为0的数量
def getPrealZeroDuo(agentNum, roleNum, groupNum, duoRank, PReal):
    duo = 0
    for i in range(len(duoRank)):
        for j in range(roleNum):
            if PReal[duoRank[i][0]][j] == 0:
                duo += 1
            if PReal[duoRank[i][1]][j] == 0:
                duo += 1

    return duo


# 计算单排负偏好为0的数量
def getPrealZeroSig(agentNum, roleNum, groupNum, duoRank, PReal):
    sig = 0
    duoRankList = []
    for i in range(len(duoRank)):
        duoRankList.append(duoRank[i][0])
        duoRankList.append(duoRank[i][1])

    for i in range(agentNum):
        for j in range(roleNum):
            if duoRankList.count(i) == 1:
                continue
            if PReal[i][j] == 0:
                sig += 1

    return sig









if __name__ == '__main__':
    agentNum = 0
    AnalData = []
    # 分析数据使用
    duo = 0
    sig = 0

    # 删除的弃局玩家
    delRowDuo = []
    delRowSig = []

    agentNum = 100
    # 双排玩家低于80%
    # duoNum = int(agentNum / 5)
    duoNum = int(agentNum * 0.3)
    roleNum = 5
    # 组的数量
    groupNum = int(agentNum / roleNum)
    L = [groupNum for i in range(roleNum)]

    #重复试验

    Q = getQMat(agentNum, roleNum)
    duoRank = getDRVec(agentNum, duoNum)
    PAll = getPMatRoundUnequal(agentNum, roleNum, duoRank)

    # PREALN0是和PREAL基本一致只差了双排重做的数据 但是弃局没清0 P没有用
    P = PAll[0]
    PRealN0 = PAll[1]
    PReal = PAll[2]
    PIndex = PAll[3]

    # 查验最佳弃局点
    # P1Max = 0
    # P1MaxIndex = [0, 0]
    # P2Max = 0
    # P2MaxIndex = [0, 0]
    # PAllMax = 0
    # PAllMaxIndex = [0, 0]
    # x1 = []
    # y1 = []
    # z1 = []
    # z2 = []
    # give = []
    # giveDuo = []
    # P1 = 0.40
    # P2 = 0.20
    # for K1 in range(20):
    #     P1 += 0.01
    #     P2 = 0.20
    #     for K2 in range(20):
    #         P2 += 0.01
    #         PREALSIG = np.zeros((agentNum, roleNum))
    #         for i in range(agentNum):
    #             for j in range(roleNum):
    #                 PREALSIG[i][j] = PREAL[i][j]
    #         for i in range(agentNum):
    #             for j in range(roleNum):
    #                 if (PREALSIG[i][j] < P1) and (PIndex[i][j] == 1):
    #                     PREALSIG[i][j] = P1
    #                 if (PREALSIG[i][j] < P2) and (PIndex[i][j] == 2):
    #                     PREALSIG[i][j] = P2

    Qp = Q * PReal
    # Qp = Q
    Qp = getNormalizedQ(Qp)
    F = [0 for i in range(agentNum)]
    start = time.perf_counter()
    T, status, p = Assignment.KM(Qp, Q, L, agentNum, roleNum, groupNum, duoRank)
    total = time.perf_counter() - start
    print("Total time:", total)
    print(p)
    TMat = np.array(T)

    PosData = analTMat(agentNum, roleNum, groupNum, TMat, PIndex, duoRank)
    MD = calGroupMD(Q, agentNum, roleNum, groupNum, TMat)

    sa = getSatisfactionNum(agentNum, roleNum, groupNum, duoRank, PRealN0, PReal, PIndex, TMat, total, MD)

    delRowDuo = detectDuoFail(roleNum, groupNum, duoRank, PIndex, PReal, TMat)
    delRowSig = detectSigFail(agentNum, roleNum, groupNum, duoRank, PIndex, PReal, TMat)
    duo = getPrealZeroDuo(agentNum, roleNum, groupNum, duoRank, PReal)
    sig = getPrealZeroSig(agentNum, roleNum, groupNum, duoRank, PReal)
    print("%d group of DuoRank teams give up the game, counts %d percent of teams" % (
    len(delRowDuo), 2 * len(delRowDuo) / duoNum * 100))
    print(
        "%d single agent give up, counts %d percent of all agents" % (len(delRowSig), len(delRowSig) / agentNum * 100))
    print("above all, we have %d agents give up in %d agents at all" % (len(delRowDuo) * 2 + len(delRowSig), agentNum))

    # 数据放入列表
    AnalData.append([agentNum, PosData[0], PosData[1], PosData[2], PosData[3], sig, duo, len(delRowSig), len(delRowDuo) * 2, MD, p, total, sa])


    AnalData = np.array(AnalData)

    # 写入Excel
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('数据', cell_overwrite_ok=True)
    colEx = ('人数', '第一偏好占比', '第二偏好占比', '无偏好占比', '负偏好占比', '单人选中负偏好弃局人数', '双排选中负偏好弃局人数', '单人弃局数', '组放弃人数', '平均差','总性能', '总时间', '满意度')
    for i in range(len(colEx)):
        sheet.write(0, i, colEx[i])
    for i in range(len(AnalData)):
        data = AnalData[i]
        for j in range(len(colEx)):
            sheet.write(i + 1, j, data[j])
    savepath = 'AnalData.xls'
    book.save(savepath)



    # #删除双排被指派了负偏好的agent
    # delRow = detectDuoFail(agentNum, roleNum, duoRank, PREAL, TMat)
    # giveDuo.append(len(delRow))
    # delRowSig = detectSigFail(agentNum, roleNum, duoRank, PIndex, TMat)
    # give.append(len(delRowSig))
    #

    # if PosData[0] > P1Max:
    #     P1Max = PosData[0]
    #     P1MaxIndex = [P1, P2]
    # if PosData[1] > P2Max:
    #     P2Max = PosData[1]
    #     P2MaxIndex = [P1, P2]
    # if (PosData[0] + PosData[1]) > PAllMax:
    #     PAllMax = PosData[0] + PosData[1]
    #     PAllMaxIndex = [P1, P2]


# 绘图
#     x = x1
#     y = give
#     fig = plt.figure(figsize=(20, 40))  # 创建图片
#     ax1 = fig.add_subplot(2, 3, 1)  # 创建子图
#     plt.plot(x, y, 'ko--')  # 在子图上画折线图，k是黑色，o是标记是圈，--是虚线
#     plt.title('quit person')
#     plt.xlim([0.4, 0.6])  # 设置X刻度范围
#     print(plt.ylim())  # 获取Y刻度范围
#     plt.show()


    # print("%d duo players and %d single players choose to give up playing games, we have %d players at all" %(len(delRow), len(delRowSig), agentNum))
    # if delRow != []:
    #     Q = np.delete(Q, delRow, axis=0)
    #     Qp = np.delete(Qp, delRow, axis=0)
    #     P = np.delete(P, delRow, axis=0)
    #     TMat = np.delete(TMat, delRow, axis=0)
    #     # 更新矩阵数据
    #     agentNum = len(Q)
    #     roleNum = len(Q[0])
    #     print("delete %d group of duoRank players" %(len(delRow)))




