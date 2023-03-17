
import numpy as np
import pulp as pl
import random
import time
import xlwt



# Q矩阵范围值常量
QRANGELOW = 400
QRANGEHIGH = 600


class Assignment:
    @classmethod
    # maxDrone: the amount of the drones
    def KM(cls, Q, L, La):
        row = len(Q)
        col = len(Q[0])

        # build a optimal problem
        pro = pl.LpProblem('Max(connection and coverage)', pl.LpMaximize)


        # build variables for the optimal problem
        lpvars = [[pl.LpVariable("x" + str(i) + "y" + str(j), lowBound=0, upBound=1, cat='Integer')  for j in range(col)]
                  for i in range(row)]

        # build optimal function
        all = pl.LpAffineExpression()
        for i in range(0, row):
            for j in range(0, col):
                all += Q[i][j] * lpvars[i][j]

        pro += all

        # build constraint for each role Role L向量约束
        LSum = 0
        for j in range(col):
            for i in range(row):
                LSum += lpvars[i][j]
            pro += LSum == L[j]
            LSum = 0

        # build constraint for each agent 可以换La
        ASum = 0
        for i in range(row):
            for j in range(col):
                ASum += lpvars[i][j]
            pro += ASum == 1
            ASum = 0

        status = pro.solve()
        print("Assignment Status: ", pl.LpStatus[status])
        print("Final Assignment Result", pl.value(pro.objective))

        # get the result of T matrix
        T = [[lpvars[i][j].varValue for j in range(col)] for i in range(row)]
        return [T, pl.value(pro.status), pl.value(pro.objective)]

# 随机化Q矩阵 V1.3
def getQMat(agentNum, roleNum):
    Q = np.random.randint(QRANGELOW,QRANGEHIGH,[agentNum, roleNum])
    Q = Q.astype(float)
    for i in range(agentNum):
        for j in range(roleNum):
            Q[i, j] = Q[i, j] / 1000
    return Q


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

    # 分析双排的所在组
    for i in range(len(duoRank)):
        for j in range(roleNum):
            for k in range(groupNum):
                if TMat[duoRank[i][0]][j][k] == 1:
                    print("%d pos in %d " % (duoRank[i][0], k))
                if TMat[duoRank[i][1]][j][k] == 1:
                    print("%d pos in %d " % (duoRank[i][1], k))


    return  PosData


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
    allAve = allAve / agentNum

    for k in range(groupNum):
        for i in range(agentNum):
            for j in range(roleNum):
                if TMat[i][j][k] == 1:
                    aveSum += Q[i][j]
        groupAve.append(aveSum / 5)
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



if __name__ == '__main__':
    # 分析数据使用
    duo = 0
    sig = 0
    AnalData = []
    # 删除的弃局玩家
    delRowDuo = []
    delRowSig = []

    agentNum = 200
    # 双排玩家占所有玩家20%
    duoNum = int(agentNum / 5)
    roleNum = 5
    # 组的数量
    groupNum = int(agentNum / roleNum)
    L = [groupNum for i in range(roleNum)]



    start = time.perf_counter()
    T, status, p = Assignment.KM(Q, L)
    total = time.perf_counter() - start
    print("Total time:", total)
    print(p)
    TMat = np.array(T)


    # 数据放入列表
    AnalData.append([p, total])


    AnalData = np.array(AnalData)

    # 写入Excel
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('数据', cell_overwrite_ok=True)
    colEx = ('总性能', '总时间')
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
    #     Q = np.delete(Q, delRow, axis=0)
    #     P = np.delete(P, delRow, axis=0)
    #     TMat = np.delete(TMat, delRow, axis=0)
    #     # 更新矩阵数据
    #     agentNum = len(Q)
    #     roleNum = len(Q[0])
    #     print("delete %d group of duoRank players" %(len(delRow)))




