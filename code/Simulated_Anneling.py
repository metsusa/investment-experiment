import numpy as np
#--Metropolis:
# if dt<0 accepted,
# else if f(dt)>rand(0,1):accpeted
#       else:refuse
def Metropolis(delta_t,T,K):        
    if delta_t<=0:
        print("error:the delta t should be accepted!")
        return False
    else:
        acpt_odds=np.exp(delta_t/(K*T))
        rdseed=np.random.random(0,1)
        if acpt_odds>rdseed:
            return True
        else: 
            return False

def costFunc(T,exp_T):      #cost function
    row,column=T.size()
    cost=0
    cost+=(T[:row,:column]-exp_T[:row,:column])^2
    return cost

def endWhile(T, exp_T):
    cost=costFunc(T,exp_T)
    if cost<0.05:
        return True
    else:
        return False

def randFlfunc(_3coef):     #fluctuation function
    alpha,beta,gamma=_3coef
    alpha+=np.random.rand(0,0.05)
    beta+=np.random.rand(0,0.05)
    gamma+=np.random.rand(0,0.05)
    return alpha,beta,gamma

def Sa(_3Randcoef,randFlfunc,GMRA,exp_T,K,costFunc):
    
    count=0
    
    _3coef=randFlfunc(_3Randcoef)     #rand fluctuation
    T=GMRA(_3coef)    #reset three tuple

    costTold=costTnew       #old one is uninitialized
    costTnew=costFunc(T,exp_T)
    deltaT=costTnew-costTold

    if Metropolis(deltaT,K)==True:
        count+=1
        if count>=10000:
            if endWhile():
                return _3coef
            else:
                T*=K        #reduce the temporary
    else:
        count=0

    # T=np.random.randint(0,2)
    
def f(_3coef):
    a,b,c=_3coef
    return a^2+2*b+c  

def cftest(T,exp_T):
    return (T-exp_T)^2

_3coef=Sa([0.5,0.5,0,5],randFlfunc,f,1.95,0.9,cftest)
print(_3coef)