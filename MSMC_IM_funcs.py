#!/usr/bin/env python3

import math
import bisect
import numpy as np
from numpy import linalg as LA
from scipy import integrate
# from scipy.linalg import expm
# from scipy.linalg import expm2

def read_lambdas_from_MSMC(fn, mu=1.25e-08): #fn should be the combined msmc output with six columns inlcuding lambda_11, lambda_12, lambda_22. 
    time_boundaries = []
    lambdas_11 = []
    lambdas_12 = []
    lambdas_22 = []
    f = open(fn, "rt")
    next(f)
    for line in f:
        fields = line.strip().split()
#        i = int(fields[0]) #The first column in msmc output is time index
        tLeft = float(fields[1]) / mu
        tRight = float(fields[2]) / mu
        lambda_11_ = float(fields[3]) * mu
        lambda_12_ = float(fields[4]) * mu
        lambda_22_ = float(fields[5]) * mu
        time_boundaries.append((tLeft, tRight))
        lambdas_11.append(lambda_11_)
        lambdas_12.append(lambda_12_)
        lambdas_22.append(lambda_22_)
    if time_boundaries[0][0] != 0 or time_boundaries[-1][1] != float('inf'):
        print("Warning! The time segment is not start from 0 or end with infinity. The program will force the time start from 0 and end with infinity now!")
        time_boundaries[0]=(0,time_boundaries[0][1])
        time_boundaries[-1]=(time_boundaries[-1][0], float('inf'))
    return time_boundaries, lambdas_11, lambdas_12, lambdas_22

def read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas): #Find the TMRCA at a certain time point (T_i) from MSMC output!! 
    tmrca_dist = []
    for i in range(len(T_i)):
        if i == 0:
            left_index = 0
        else:
            left_index = bisect.bisect_right(left_boundaries, T_i[i]) - 1  #left_index -- the nearest time boundary at time t
        tleft = left_boundaries[left_index]
        lambda_ = lambdas[left_index] #the value of lambda at the time point 't'
        if left_index==0:
            delta = T_i[i] - tleft
            integ = delta * lambdas[0]
            tmrca = lambda_ * math.exp(-integ)
        else:
            deltas = [left_boundaries[j+1] - left_boundaries[j] for j in range(left_index)] #all time intervals in time t
            deltas.append(T_i[i] - tleft)
            integ = sum(delta * lambda_prime for delta, lambda_prime in zip(deltas, lambdas[:left_index+1]))
            tmrca = lambda_ * math.exp(-integ)
        tmrca_dist.append(tmrca)
    return tmrca_dist
    
def makeQ(m, N1, N2): #The matrix includes migration. The sum of each row is 0 in the matrix
     q = np.matrix([
         [-(2*m+1/(2*N1)), 2 * m, 0, 1/(2*N1), 0],
         [m, -(m+m), m, 0, 0],
         [0, 2*m, -(2*m+1/(2*N2)), 0 , 1/(2*N2)],
         [0, 0, 0, -m, m],
         [0, 0, 0, m, -m]])
     return q

# def makeQexp(qMatrix, t): #Make matrix exponential during timw interval delta_t. TIP: expm, expm2, expm3 give results in numpy.array format.
#    qexp = expm(qMatrix*t)
#    return qexp

def makeQexp(qMatrix, t):
    evalue, evector = LA.eig(qMatrix*t) #matrix diagonalization, which is basically internal calculation in expm() function
    qexp = np.asarray(evector * np.diag(np.exp(evalue)) * LA.inv(evector))
    return qexp
        
def makeQpropagator_xvector_Symmlist(x_0, time_boundaries, N1, N2, m): 
    x_temp = x_0
    List_x_vector = [np.asarray(x_0)]
    for i in range(1,len(time_boundaries)):
        q = makeQ(m[i], N1[i], N2[i])       
        x_temp = np.dot(x_temp, makeQexp(q, time_boundaries[i]-time_boundaries[i-1]))
        List_x_vector.append(x_temp)
    return List_x_vector

def computeTMRCA_t0_DynamicN_caltbound_mlist(t, List_x_vector, time_boundaries, N1, N2):
    t_index = bisect.bisect_right(time_boundaries, t) - 1
    x_t = List_x_vector[t_index]
    p = x_t[0]/(2. * N1[t_index]) + x_t[2]/(2. * N2[t_index])
    return p
    
def cal_tmrca_IM_Symmlist(x_0, time_boundaries, N1, N2, m, times):
    List_x_vector = makeQpropagator_xvector_Symmlist(x_0, time_boundaries, N1, N2, m)
    tmrca_dist = [computeTMRCA_t0_DynamicN_caltbound_mlist(t, List_x_vector, time_boundaries, N1, N2) for t in times]
    return tmrca_dist

def F_computeTMRCA_t0_DynamicN_caltbound_mlist(t, List_x_vector, time_boundaries, N1, N2): #Antiderivative function of P(tMRCA)
    F_t, err = integrate.quad(computeTMRCA_t0_DynamicN_caltbound_mlist, 0, t, args=(List_x_vector, time_boundaries, N1, N2), limit=1000)
    return F_t, err

def cal_coalescence_rate_bytmrca_mlist(t, List_x_vector, time_boundaries, N1, N2): 
    #Hazard Function: cross_lambda = P(tmrca)/(1-F(tmrca)) where F(tmrca) is the Antiderivative of P(tmrca)
    P_t = computeTMRCA_t0_DynamicN_caltbound_mlist(t, List_x_vector, time_boundaries, N1, N2)
    F_t, err = F_computeTMRCA_t0_DynamicN_caltbound_mlist(t, List_x_vector, time_boundaries, N1, N2)
    lambda_t = P_t / (1 - F_t)   
    return lambda_t
    
def Chi_Square_Mstopt0_DynamicN_Symmlist(Params, beta, time_boundaries, times, realTMRCA):
    length = len(time_boundaries)
    N1 = Params[0:length]
    N2 = Params[length: 2*length]
    m = Params[2*length:]
    Chi_Square = []
    for lambda_index, x_0 in zip([0, 1, 2], [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]]):
        chi_square = 0
        computedTMRCA = cal_tmrca_IM_Symmlist(x_0, time_boundaries, N1, N2, m, times)
        for i in range(len(times)):
#            if math.isnan(computedTMRCA[i]):
#                raise Exception("Theoretical tMRCA distribtuion is not a number")
            if math.isnan(computedTMRCA[i]) or realTMRCA[lambda_index][i] == 0:
                continue
            chi_square+=(realTMRCA[lambda_index][i]-computedTMRCA[i])**2/realTMRCA[lambda_index][i]
        Chi_Square.append(chi_square)
    b1 = beta[0]
    b2 = beta[1]
    penalty1 = b1 * (sum([m[i]*(time_boundaries[i+1]-time_boundaries[i]) for i in range(0, length-1)]) + m[-1]*time_boundaries[-1]*3) #penalty on migrate rate in each time segment
    penalty2 = b2 * sum([abs((n1-n2)/(n1+n2)) for n1,n2 in zip(N1,N2)]) #penalty of absolute function over (N1-N2)/(N1+N2)
#    penalty2 = b2 * sum([((n1-n2)/(n1+n2))**2 for n1,n2 in zip(N1,N2)]) #penalty of sqaure over (N1-N2)/(N1+N2)
    total_chi_square = sum(Chi_Square) + penalty1 + penalty2
#    if 1 - math.exp(-2 * m[0] * time_boundaries[0]) > 0.99: total_chi_square = 1e500
    return total_chi_square

def scaled_chi_square_Mstopt0_DynamicN_Symmlist(Params, beta, time_boundaries, times, realTMRCA, repeat, segs):
    #Here migration rate m is read as a list instead of a constant value, corresponding to our continuous/dynamic migration rate model   
    N1_ = []
    N2_ = []
    m_ = []
    uniqN1 = Params[:sum(segs)]
    uniqN2 = Params[sum(segs): 2*sum(segs)]
    uniqm = Params[2*sum(segs):]
    for i in range(len(segs)):
        N1_ = N1_ + np.repeat(uniqN1[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
        N2_ = N2_ + np.repeat(uniqN2[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
        m_ = m_ + np.repeat(uniqm[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
    if max(N1_) > math.log(1e7) or max(N2_) > math.log(1e7) or min(N1_) <= 0 or min(N2_) <= 0 or max(m_) > math.log(100): #Setting limit for m here just to avoid overflow issue
        chi_square_score = 1e500
    else:
        unscale_list = [[math.exp(n1) for n1 in N1_], [math.exp(n2) for n2 in N2_], [math.exp(m) for m in m_]]
        unscale_pars = [value for sublist in unscale_list for value in sublist]
        chi_square_score = Chi_Square_Mstopt0_DynamicN_Symmlist(unscale_pars, beta, time_boundaries, times, realTMRCA)
    return chi_square_score

def cumulative_Symmigproportion(time_boundaries, m): #Here m is a list whose length is consistent with the length the time_boundaries(right_boundaries, which is not start with 0)!
    CDF = []
    integ = 2 * m[0] * time_boundaries[0]
    cdf = 1 - math.exp(-integ)
    CDF.append(cdf)
    for i in range(1,len(time_boundaries)):
        integ += 2* m[i] * (time_boundaries[i]-time_boundaries[i-1])
#        print(i, m[i], time_boundaries[i]-time_boundaries[i-1], integ)
        cdf = 1 - math.exp(-integ)
        CDF.append(cdf)
    return CDF

def getCDFintersect(left_boundaries, right_boundaries, CDF, val):
    xVec = [(left_time_boundary + right_time_boundary)/2 for left_time_boundary, right_time_boundary in zip(left_boundaries, right_boundaries)]
    yVec = CDF
    i = 0
    if yVec[0] < val:
        while yVec[i] < val:
            i += 1
        assert i > 0 and i <= len(yVec), "CDF intersection index out of bounds: {}".format(i)
        assert yVec[i - 1] < val and yVec[i] >= val, "this should never happen"
        intersectDistance = (val - yVec[i - 1]) / (yVec[i] - yVec[i - 1])
        CDFintersect = xVec[i - 1] + intersectDistance * (xVec[i] - xVec[i - 1])
    else:
        CDFintersect = val/yVec[0] * xVec[0]
    return CDFintersect



############################ Asymmetric migration mode #############################################################################################################

def makeQ2m(m1, m2, N1, N2): #The matrix includes migration. The sum of each row is 0 in the matrix
     q = np.matrix([
         [-(2*m1+1/(2*N1)), 2*m1, 0, 1/(2*N1), 0],
         [m2, -(m1+m2), m1, 0, 0],
         [0, 2*m2, -(2*m2+1/(2*N2)), 0 , 1/(2*N2)],
         [0, 0, 0, -m1, m1],
         [0, 0, 0, m2, -m2]])
     return q

def makeQpropagator_xvector_2mlist(x_0, time_boundaries, N1, N2, m1, m2): 
    #Here migration rate m is read as a list instead of a constant value, corresponding to our continuous/dynamic migration rate model   
    #The following two lines artificially force t0 and t0_index to be 0 instead of read in as a variable
    x_temp = x_0
    List_x_vector = [np.asarray(x_0)]
    for i in range(1,len(time_boundaries)):
        q = makeQ2m(m1[i], m2[i], N1[i], N2[i])
        x_temp = np.dot(x_temp, makeQexp(q, time_boundaries[i]-time_boundaries[i-1]))
        List_x_vector.append(x_temp)
    return List_x_vector

def cal_tmrca_IM_2mlist(x_0, time_boundaries, N1, N2, m1, m2, times):
    List_x_vector = makeQpropagator_xvector_2mlist(x_0, time_boundaries, N1, N2, m1, m2)
    tmrca_dist = [computeTMRCA_t0_DynamicN_caltbound_mlist(t, List_x_vector, time_boundaries, N1, N2) for t in times]
    return tmrca_dist

def cumulative_2migproportion(time_boundaries, m1, m2): #Here m is a list whose length is consistent with the length the time_boundaries(right_boundaries, which is not start with 0)!
    CDF = []
    integ = (m1[0]+m2[0]) * time_boundaries[0]
    cdf = 1 - math.exp(-integ)
    CDF.append(cdf)
    for i in range(1,len(m1)):
        integ += (m1[i]+m2[i])  * (time_boundaries[i]-time_boundaries[i-1])
#        print(i, m[i], time_boundaries[i]-time_boundaries[i-1], integ)
        cdf = 1 - math.exp(-integ)
        CDF.append(cdf)
    return CDF
    
def read_tmrcavalue_from_MSMC(t, left_boundaries, lambdas): #Find the TMRCA at a certain time point (T_i) from MSMC output!! 
    left_index = bisect.bisect_right(left_boundaries, t) - 1  #left_index -- the nearest time boundary at time t
    tleft = left_boundaries[left_index]
    lambda_ = lambdas[left_index] #the value of lambda at the time point 't'
    if left_index==0:
        delta = t - tleft
        integ = delta * lambdas[0]
        tmrca = lambda_ * math.exp(-integ)
    else:
        deltas = [left_boundaries[j+1] - left_boundaries[j] for j in range(left_index)] #all time intervals in time t
        deltas.append(t - tleft)
        integ = sum(delta * lambda_prime for delta, lambda_prime in zip(deltas, lambdas[:left_index+1]))
        tmrca = lambda_ * math.exp(-integ)
    return tmrca
