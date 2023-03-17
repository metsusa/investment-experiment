import matplotlib as mp
import random
import pulp as pl
from gurobipy import *

MAXAGENT=100
MAXROLE=100

class Assigment:
    def GMRA(Qp,Q,L,alpha,beta,gamma):
        row,column=Qp

        pbm=pl.LpProblem('Max(connection and coverage)', pl.LpMaximize)
        slv=pl.getsolver('GUROBI_CMD')

        lp_xvar = [[pl.LpVariable("x" + str(i) + "y" + str(j), lowBound=0, upBound=1, cat='Integer')
                    for j in  range(column)]
                    for i in range(row)]

        all=pl.AffineExpression([[(lp_xvar[i][j],Qp[i][j])
                                  for i in range(column)]
                                  for j in range(row)])
        
    def persentDegreeed(cls,alpha,beta,gamma):
        # T[0]*=alpha
        # T[1]*=beta
        # T[2]*=gamma
        
