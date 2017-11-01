#!/usr/bin/env python3.4

import MSMC_IM_funcs
import argparse
import math
import bisect
import numpy as np
from multiprocessing import Pool, Process
from scipy.linalg import expm
from scipy.optimize import fmin_powell

parser = argparse.ArgumentParser(description='Find parameters with the max likelihood for fitting IM model to MSMC')
parser.add_argument("Input", help="OUTPUT from MSMC")
parser.add_argument("-N1", default=15000, type=float, help="Effective population size of Pop1. Default=1500")
parser.add_argument("-N2", default=15000, type=float, help="Effective population size of Pop2. Default=1500")
parser.add_argument("-NA", default=15000, type=float, help="Effective population sieze of a population ancestral to population 1 and population 2. Default=1500")
parser.add_argument("-T", default=2000, type=float, help="Split time between population 1 and population 2. Default=2000")
parser.add_argument("-t0", default=500, type=float, help="Time when migrations between population 1 and population 2 stop. Default=500")
parser.add_argument("-m", default=10e-5, type=float, help="Symetric Migration rate. Default=10e-5")
parser.add_argument("--Ti_MSMC", default=False, action="store_true", help="whether use the same time boundaries from MSMC for fitting(recommended).")
parser.add_argument("-n_T", default=1000, type=int, help="Number of time segments in total in fitting. Default=1000")
parser.add_argument("-N0", default=20000, type=float, help="Average effective population size. Default=20000") 
args = parser.parse_args()

time_lr_boundaries, lambdas_11, lambdas_12, lambdas_22 = MSMC_IM_funcs.read_lambdas_from_MSMC(args.Input) #time_lr_boundaries=[[left1,right1], [left2,right2], [] ... []]
left_boundaries = [k[0] for k in time_lr_boundaries] 
time_boundaries = [k[0] for k in time_lr_boundaries]
time_boundaries.append(time_boundaries[-1] * 4) #time_boundaries = all left_boundaries plus the last right time boundary.
if args.Ti_MSMC:
    T_i = [(k[0]+k[1])/2 for k in time_lr_boundaries[:-1]]
    T_i.append(time_lr_boundaries[-1][0]*2.5) 
else:
    T_i = [-math.log(1 - i/args.n_T) * 2 * args.N0 for i in range(args.n_T)] 

realTMRCA_11 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_11)
realTMRCA_12 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_12)
realTMRCA_22 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_22)
realTMRCA = np.array((realTMRCA_11, realTMRCA_12, realTMRCA_22))


'''
### Use the results from minimization as initial parameters for the next fitting to speed up
All_Init_chisquare=[]
All_Final_chisquare=[]
All_T_prior=[]  
All_best_params=[]
T_indice = 0
Tbounds = [time_boundaries[T_indice], time_boundaries[T_indice+1]]
T_prior = (Tbounds[0]+Tbounds[1])/2
t0_prior = (Tbounds[0]+Tbounds[1])/4
init_N1 = [math.log(args.N1)]
init_N2 = [math.log(args.N2)]
init_NA = [math.log(args.NA)]*len(left_boundaries)
init_m = [10000*math.atanh(2e3*args.m-1)]
init_t0 = [math.log(t0_prior)]
init_params = [value for sublist in [init_N1, init_N2, init_NA, init_m, init_t0] for value in sublist]

for T_indice in range(len(left_boundaries)): 
    Tbounds = [time_boundaries[T_indice], time_boundaries[T_indice+1]] 
    T_prior = (Tbounds[0]+Tbounds[1])/2
    t0_prior = (Tbounds[0]+Tbounds[1])/4
    init_chisquare = scaled_chi_square_Mstopt0_DynamicN(init_params, Tbounds, T_indice, left_boundaries, T_i, realTMRCA)
    Scaled_Params = fmin_powell(scaled_chi_square_Mstopt0_DynamicN, init_params, args=(Tbounds, T_indice, left_boundaries, T_i, realTMRCA), xtol=1e-4, ftol=1e-8)
    final_chisquare = scaled_chi_square_Mstopt0_DynamicN(Scaled_Params, Tbounds, T_indice, left_boundaries, T_i, realTMRCA)
#    print(T_indice, t0_prior, T_prior, Tbounds, init_chisquare, final_chisquare)init_N1 = [math.log(args.N1)]
# Use the results from minimization as initial parameters for the next fitting to speed up
    init_N1 = Scaled_Params[:T_indice+1].append(math.log(args.N1))
    init_N2 = Scaled_Params[T_indice+1: 2*T_indice+2].append(math.log(args.N2))
    init_NA = Scaled_Params[2*best_T_indice+2: -2][1:]
    init_m = Scaled_Params[-2]
    init_t0 = Scaled_Params[-1]
    init_params = [value for sublist in [init_N1, init_N2, init_NA, init_m, init_t0] for value in sublist]
    All_T_prior.append(T_prior)
    All_best_params.append(Scaled_Params)
    All_Init_chisquare.append(init_chisquare)
    All_Final_chisquare.append(final_chisquare)
'''
###RUN fmin_powell minimization in parallel instead of in a for loop to SPEED UP
def loop_minimization(T_indice):
    if T_indice < len(time_boundaries):
        Tbounds = [time_boundaries[T_indice], time_boundaries[T_indice+1]] 
        T_prior = (Tbounds[0]+Tbounds[1])/2
        t0_prior = (Tbounds[0]+Tbounds[1])/4
        par_list=[[math.log(args.N1)]*(T_indice+1), [math.log(args.N2)]*(T_indice+1), [math.log(args.NA)]*(len(left_boundaries)-T_indice), [10000*math.atanh(2e3*args.m-1)], [math.log(t0_prior)]]
        init_params = [value for sublist in par_list for value in sublist]
        init_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN(init_params, Tbounds, T_indice, left_boundaries, T_i, realTMRCA)
        Scaled_Params = fmin_powell(MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN, init_params, args=(Tbounds, T_indice, left_boundaries, T_i, realTMRCA), xtol=1e-4, ftol=1e-8)
        final_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN(Scaled_Params, Tbounds, T_indice, left_boundaries, T_i, realTMRCA)
        return T_prior, Scaled_Params, init_chisquare, final_chisquare
    else: 
        raise Exception("T is out of range")

if __name__ == '__main__':
    pool = Pool()
    time_lr_boundaries, lambdas_11, lambdas_12, lambdas_22 = MSMC_IM_funcs.read_lambdas_from_MSMC(args.Input)
    left_boundaries = [k[0] for k in time_lr_boundaries]
#    processess1=[] 
#     for i in [0,1,2]:
#        processess1.append(Process(loop_minimization,args=(i,)))        
#   pool.map(processess1)            
    #print(pool.map(loop_minimization, [0,1,2]))
    #return_results = pool.map(loop_minimization, range(3))
    return_results = pool.map(loop_minimization, range(len(left_boundaries)))
    All_Init_chisquare=[]
    All_Final_chisquare=[]
    All_T_prior=[]  
    All_best_params=[]
    for results in return_results:
        T_prior = results[0]
        Scaled_Params = results[1]
        init_chisquare = results[2]
        final_chisquare = results[3]
        All_T_prior.append(T_prior)
        All_best_params.append(Scaled_Params)
        All_Init_chisquare.append(init_chisquare)
        All_Final_chisquare.append(final_chisquare)
      
'''
All_Init_chisquare=[]
All_Final_chisquare=[]
All_T_prior=[]  
All_best_params=[]
#for T_indice in range(len(left_boundaries)):  #len(left_boundaries) is the total number of time intervals
if True:
    T_indice=1
    Tbounds = [time_boundaries[T_indice], time_boundaries[T_indice+1]] 
    T_prior = (Tbounds[0]+Tbounds[1])/2
    t0_prior = (Tbounds[0]+Tbounds[1])/4
    par_list=[[math.log(args.N1)]*(T_indice+1), [math.log(args.N2)]*(T_indice+1), [math.log(args.NA)]*(len(left_boundaries)-T_indice), [10000*math.atanh(2e3*args.m-1)], [math.log(t0_prior)]]
    init_params = [value for sublist in par_list for value in sublist]
    init_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN(init_params, Tbounds, T_indice, left_boundaries, T_i, realTMRCA)
    Scaled_Params = fmin_powell(MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN, init_params, args=(Tbounds, T_indice, left_boundaries, T_i, realTMRCA), xtol=1e-4, ftol=1e-6)
    final_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN(Scaled_Params, Tbounds, T_indice, left_boundaries, T_i, realTMRCA)
##    print(T_indice, t0_prior, T_prior, Tbounds, init_chisquare, final_chisquare)
    All_T_prior.append(T_prior)
    All_best_params.append(Scaled_Params)
    All_Init_chisquare.append(init_chisquare)
    All_Final_chisquare.append(final_chisquare)
    
best_T_indice=T_indice; Best_Scaled_Params=Scaled_Params
'''
best_T_indice=All_Final_chisquare.index(min(All_Final_chisquare))
Best_Scaled_Params=All_best_params[best_T_indice]

N1_List = [math.exp(n1) for n1 in Best_Scaled_Params[:best_T_indice+1]]
N2_List = [math.exp(n2) for n2 in Best_Scaled_Params[best_T_indice+1: 2*best_T_indice+2]]
NA_List = [math.exp(na) for na in Best_Scaled_Params[2*best_T_indice+2: -2]]

print("T_indice:{}".format(best_T_indice),
    "N1:{}".format(N1_List),
    "N2:{}".format(N2_List),
    "NA:{}".format(NA_List),
    "m:{}".format((math.tanh(Best_Scaled_Params[-2]/10000)+1)/2e3), 
    "t0:{}".format(math.exp(Best_Scaled_Params[-1])), sep="\n")

print("time_index", "left_time_boundary", "right_time_boundary", "N1", "N2", "NA", sep="\t")#, "mid_T", "Init-Chi-Square", "Final-Chi-Square", sep="\t")
for i in range(len(left_boundaries)):
    if i < best_T_indice:
        print(i, time_boundaries[i], time_boundaries[i+1], N1_List[i], N2_List[i], "NA", sep="\t")#, All_T_prior[i], All_Init_chisquare[i], All_Final_chisquare[i], sep="\t")
    elif i ==  best_T_indice:
        print(i, time_boundaries[i], time_boundaries[i+1], N1_List[i], N2_List[i], NA_List[i-best_T_indice], sep="\t")#,  All_T_prior[i], All_Init_chisquare[i], All_Final_chisquare[i], sep="\t")
    elif i > best_T_indice:
        print(i, time_boundaries[i], time_boundaries[i+1], "NA", "NA", NA_List[i-best_T_indice], sep="\t")#, All_T_prior[i], All_Init_chisquare[i], All_Final_chisquare[i], sep="\t")
