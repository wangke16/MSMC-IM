#!/usr/bin/env python3.4

import math
import bisect
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fmin_powell

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
         [-(2*m+1/(2*N1)), 2*m, 0, 1 / (2*N1), 0],
         [m, -(m+m), m, 0, 0],
         [0, 2*m, -(2*m+1/(2*N2)), 0 , 1/(2*N2)],
         [0, 0, 0, -m, m],
         [0, 0, 0, m, -m]])
     return q
     
def makeQ_0(N1, N2): #The matrix when migration is 0 (symmetric migrations stop when t<t0)
     q = np.matrix([
         [-1/(2*N1), 0, 0, 1/(2*N1), 0],
         [0, 0, 0, 0, 0],
         [0, 0, -1/(2*N2), 0 , 1/(2*N2)],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]])
     return q
     
def makeQexp(qMatrix, t): #Make matrix exponential during timw interval delta_t. 
    qexp = expm(qMatrix*t)
    return qexp
    
def computeTMRCA(N1, N2, NA, T, t, init_vector): #Compute TMRCA distribution with CONSTANT population size and WITHOUT t0.
    x_0 = init_vector
    q = makeQ_0(N1, N2)
    qexp = makeQexp(q,t)
    qexp_T = makeQexp(q,T)
    if t >= T:
        interger = (t - T) * 1./(2. * NA)
        p = (np.dot(x_0, qexp_T)[0] + np.dot(x_0, qexp_T)[1] + np.dot(x_0, qexp_T)[2])* 1. /(2. * NA) * math.exp(-interger)
    else:
        p = np.dot(x_0, qexp)[0]/(2. * N1) + np.dot(x_0, qexp)[2]/(2. * N2)
    if math.isnan(p):
        raise Exception("ComputedTMRCA is not a number when time is at {} generations".format(t)) 
    return p
    
def computeTMRCA_t0(N1, N2, NA, T, m, t, t0, init_vector): #Compute TMRCA distribution with CONSTANT population size and t0. 
    x_0 = init_vector
    q_0 = makeQ_0(N1, N2)
    q = makeQ(m, N1, N2)
    qexp_t0 = makeQexp(q_0,t0)
    x_t0 = np.dot(x_0, qexp_t0)
    if t<=t0:
        qexp = makeQexp(q_0,t)
        p = np.dot(x_0, qexp)[0]/(2. * N1) + np.dot(x_0, qexp)[2]/(2. * N2)
    elif t<T:
        qexp = makeQexp(q,t-t0)
        p = np.dot(x_t0, qexp)[0]/(2. * N1) + np.dot(x_t0, qexp)[2]/(2. * N2)
    else:
        qexp_T = makeQexp(q,T-t0)
        interger = (t - T) * 1./(2. * NA)
        p = (np.dot(x_t0, qexp_T)[0] + np.dot(x_t0, qexp_T)[1] + np.dot(x_t0, qexp_T)[2])* 1. /(2. * NA) * math.exp(-interger)
    return p
    
def makeQpropagator_beft0(init_vector, T_indice, time_boundaries, N1, N2, m, t0): 
#Make a list of Qpropagator before split time T e.g.[Q1, Q1*Q2, Q1*Q2*Q3, ... Q1*Q2*Q3*..Qt0 etc.]
#    List_Qpropagator = []
    x_0 = init_vector
    List_x_vector_beft0 = [init_vector]
    q_temp=np.zeros((5,5))
    np.fill_diagonal(q_temp,1)
    t0_indice = bisect.bisect_right(time_boundaries, t0)-1
    for i in range(t0_indice):
        q = makeQ_0(N1[i], N2[i])
        q_exm = makeQexp(q, time_boundaries[i+1]-time_boundaries[i])
        q_temp = np.dot(q_temp,q_exm) 
        x_vector = np.dot(x_0,q_temp) 
        List_x_vector_beft0.append(x_vector)
#        List_Qpropagator.append(q_temp)
    if t0 == time_boundaries[t0_indice]: #t0 can only be >=time_boundaries[t0_indice] (of course t0 < time_boundaries[t0_indice+1])
        Qpropagator_t0 = np.copy(q_temp)
    else:
        q_t0 = makeQ_0(N1[t0_indice], N2[t0_indice]) #Calculate matrix before t0 thus without migration 
        q_temp=np.dot(q_temp, makeQexp(q_t0, t0-time_boundaries[t0_indice])) #Here is the Qpropagator at t0
        Qpropagator_t0 = np.copy(q_temp) #Here is the Qpropagator at t0
    return List_x_vector_beft0, Qpropagator_t0
#        List_Qpropagator.append(Qpropagator_t0)
#    return List_Qpropagator, Qpropagator_t0
    
def makeQpropagator_aft0(init_vector, Qpropagator_t0, T_indice, time_boundaries, N1, N2, m, t0, T):
    List_x_vector=[]
    x_0 = init_vector
    x_t0 = np.dot(x_0, Qpropagator_t0)
    t0_indice = bisect.bisect_right(time_boundaries, t0)-1
    if T_indice == t0_indice: #T_indice can only be >= t0_indice because T must >=t0. 
        if T == t0:
            List_x_vector.append(x_t0)
        else: 
            q_T = makeQ(m, N1[T_indice], N2[T_indice])
            x_T = np.dot(x_t0, makeQexp(q_T, T-t0))
            List_x_vector.append(x_T)
    else:
        q_t0_prime = makeQ(m, N1[t0_indice], N2[t0_indice]) #Calculate matrix after t0 thus with migration
        x_temp = np.dot(x_t0, makeQexp(q_t0_prime, time_boundaries[t0_indice+1]-t0))
        List_x_vector.append(x_temp)
        if T_indice == t0_indice + 1: 
             q_T = makeQ(m, N1[T_indice], N2[T_indice]) #Calculate matrix during time period [time_boundaries[T_indice], T]
             x_T =  np.dot(x_temp, makeQexp(q_T, T - time_boundaries[T_indice]))
             List_x_vector.append(x_T)
        else:
            for j in range(t0_indice+1, T_indice):
                q = makeQ(m, N1[j], N2[j])
                x_temp = np.dot(x_temp, makeQexp(q, time_boundaries[j+1]-time_boundaries[j]))
                List_x_vector.append(x_temp)
            q_T = makeQ(m, N1[T_indice], N2[T_indice]) 
            x_temp =  np.dot(x_temp, makeQexp(q_T, T - time_boundaries[T_indice]))
            List_x_vector.append(x_temp)
    return List_x_vector

#Compute TMRCA distribution with DYNAMIC population size and t0. 
def computeTMRCA_t0_DynamicN(List_x_vector_beft0, List_x_vector_aft0, time_boundaries, N1, N2, NA, T, m, t, t0, init_vector):
    x_0 = init_vector
    T_indice = bisect.bisect_right(time_boundaries, T)-1
    t_indice = bisect.bisect_right(time_boundaries, t)-1
    t0_indice = bisect.bisect_right(time_boundaries, t0)-1
    if t <= t0:
        x_t = np.dot(List_x_vector_beft0[t_indice], makeQexp(makeQ_0(N1[t_indice], N2[t_indice]), t-time_boundaries[t_indice]))
        p = x_t[0]/(2. * N1[t_indice]) + x_t[2]/(2. * N2[t_indice])
    elif t < T:
        x_t = np.dot(List_x_vector_aft0[t_indice-t0_indice], makeQexp(makeQ(m, N1[t_indice], N2[t_indice]), t-time_boundaries[t_indice]))
        p = x_t[0]/(2. * N1[t_indice]) + x_t[2]/(2. * N2[t_indice])
    else:
        x_T = List_x_vector_aft0[-1]
        if t_indice - T_indice == 0:
            interger = (t - T) * 1./(2. * NA[t_indice-T_indice])
        elif t_indice - T_indice == 1:
            interger = (time_boundaries[T_indice+1] - T) * 1./(2. * NA[0]) + (t-time_boundaries[t_indice]) * 1./(2. * NA[t_indice-T_indice])
        else:
            interger = (time_boundaries[T_indice+1] - T) * 1./(2. * NA[0])
            for i in range(1,t_indice-T_indice):
                interger+=(time_boundaries[T_indice+i+1] - time_boundaries[T_indice+i]) * 1./(2. * NA[i+1])
            interger+=(t-time_boundaries[t_indice]) * 1./(2. * NA[t_indice-T_indice])
        p = (x_T[0] + x_T[1] + x_T[2]) * 1/(2*NA[t_indice-T_indice]) * math.exp(-interger)
    return p
        
##-----------------Calculate ChiSquare & Fitting tMRCA distibution from MSMC to theory & Estimating parameters from theory------------------
def Chi_Square(Params, times, realTMRCA):
    N1, N2, NA, T = Params
    Chi_Square = []
    for lambda_index, init_vector in zip([0, 1, 2], [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]]):
        computedTMRCAdist = [computeTMRCA(N1, N2, NA, T, t, init_vector) for t in times]
        chi_square = 0
        for i in range(len(times)):
            if realTMRCA[lambda_index][i] == 0:
#                    continue
                realTMRCA[lambda_index][i] = 1e-20
            chi_square+=(realTMRCA[lambda_index][i]-computedTMRCAdist[i])**2/realTMRCA[lambda_index][i]
        Chi_Square.append(chi_square)
    total_chi_square = sum(Chi_Square)
    if math.isnan(total_chi_square):
        raise Exception("Total Chi-square value is not a number")
    return total_chi_square

def scaled_chi_square(Params, times, realTMRCA):
    N1_, N2_, NA_, T_ = Params
    if N1_ > 20 or N2_ > 20 or NA_ > 20 or N1_ < 0 or N2_ < 0 or NA_ < 0:
        chi_square_score = 1e20
    else:
        chi_square_score = Chi_Square(np.array([math.exp(N1_),math.exp(N2_),math.exp(NA_),math.exp(T_)]), times, realTMRCA)
    return chi_square_score
    
def Chi_Square_Mstopt0(Params, times, realTMRCA):
    N1, N2, NA, T, m, t0= Params
    Chi_Square = []
    for lambda_index, init_vector in zip([0, 1, 2], [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]]):
        chi_square = 0
        computedTMRCAdist = [computeTMRCA_t0(N1, N2, NA, T, m, t, t0, init_vector) for t in times]
        for i in range(len(times)):
            if realTMRCA[lambda_index][i] == 0:
#                continue
                realTMRCA[lambda_index][i] = 1e-20
            chi_square+=(realTMRCA[lambda_index][i]-computedTMRCAdist[i])**2/realTMRCA[lambda_index][i]
        #chi_square = sum((realTMRCA[lambda_index][i]-computedTMRCAdist[i])**2/realTMRCA[lambda_index][i] for i in range(len(times)))
        Chi_Square.append(chi_square)
    total_chi_square = sum(Chi_Square)
    if math.isnan(total_chi_square):
        raise Exception("Total Chi-square value is not a number")
    return total_chi_square

def scaled_chi_square_Mstopt0(Params, times, realTMRCA):
    N1_, N2_, NA_, T_, m_, t0_ = Params
    if N1_ > 20 or N2_ > 20 or NA_ > 20 or N1_ < 0 or N2_ < 0 or NA_ < 0 or t0_ > T_:
        chi_square_score = 1e20
    else:
        chi_square_score=Chi_Square_Mstopt0(np.array([math.exp(N1_),math.exp(N2_),math.exp(NA_),math.exp(T_), (math.tanh(m_/10000)+1)/2e3, math.exp(t0_)]), times, realTMRCA)
    return chi_square_score
    
def Chi_Square_Mstopt0_DynamicN(Params,Tbounds, T_indice, time_boundaries, times, realTMRCA):
    N1 = Params[0:T_indice+1]
    N2 = Params[T_indice+1: 2*T_indice+2]
    NA = Params[2*T_indice+2: (len(time_boundaries)) + T_indice + 2]
    T = sum(Tbounds)/2
    m = Params[-2]
    t0 = Params[-1]
    Chi_Square = []
    for lambda_index, init_vector in zip([0, 1, 2], [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]]):
        chi_square = 0
        List_x_vector_beft0, Qpropagator_t0 = makeQpropagator_beft0(init_vector, T_indice, time_boundaries, N1, N2, m, t0)
        List_x_vector_aft0 = makeQpropagator_aft0(init_vector, Qpropagator_t0, T_indice, time_boundaries, N1, N2, m, t0, T)
        computedTMRCAdist = [computeTMRCA_t0_DynamicN(List_x_vector_beft0, List_x_vector_aft0, time_boundaries, N1, N2, NA, T, m, t, t0, init_vector) for t in times]
        for i in range(len(times)):
            if realTMRCA[lambda_index][i] == 0:
#                continue
                realTMRCA[lambda_index][i] = 1e-20
            if math.isnan(computedTMRCAdist[i]):
                continue
            chi_square+=(realTMRCA[lambda_index][i]-computedTMRCAdist[i])**2/realTMRCA[lambda_index][i]
            if math.isnan(chi_square):
                #print(realTMRCA[lambda_index][i], computedTMRCAdist[i])
                raise Exception("Chi-square value is not a number")
        Chi_Square.append(chi_square)
    total_chi_square = sum(Chi_Square)
    return total_chi_square

def scaled_chi_square_Mstopt0_DynamicN(Params, Tbounds, T_indice, time_boundaries, times, realTMRCA):
    #print(T_indice,Params)
    N1_ = Params[0:T_indice+1]
    N2_ = Params[T_indice+1: 2*T_indice+2]
    NA_ = Params[2*T_indice+2: (len(time_boundaries)) + T_indice + 2]
    m_ = Params[-2]
    t0_ = Params[-1]
    T_ = sum(Tbounds)/2
    if max(N1_) > 20 or max(N2_) > 20 or max(NA_) > 20 or min(N1_) < 0 or min(N2_) < 0 or min(NA_) < 0 or math.exp(t0_) > T_ or math.exp(t0_)<0:
        chi_square_score = 1e20
    else:
        unscale_list=[[math.exp(n1) for n1 in Params[:T_indice+1]], [math.exp(n2) for n2 in Params[T_indice+1: 2*T_indice+2]], [math.exp(na) for na in Params[2*T_indice+2: (len(time_boundaries))+T_indice+2]], [(math.tanh(Params[-2]/10000)+1)/2e3], [math.exp(Params[-1])]]
        unscale_pars=[value for sublist in unscale_list for value in sublist]
        chi_square_score=Chi_Square_Mstopt0_DynamicN(unscale_pars, Tbounds, T_indice, time_boundaries, times, realTMRCA)
    return chi_square_score