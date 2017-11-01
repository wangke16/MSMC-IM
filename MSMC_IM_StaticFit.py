#!/usr/bin/env python3.4

import MSMC_IM_funcs
import argparse
import math
import bisect
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fmin_powell

parser = argparse.ArgumentParser(description='Find parameters with the max likelihood for fitting IM model to MSMC')
parser.add_argument("Input", help="OUTPUT from MSMC")
parser.add_argument("-N1", default=15000, type=float, help="Effective population size of Pop1. Default=1500")
parser.add_argument("-N2", default=15000, type=float, help="Effective population size of Pop2. Default=1500")
parser.add_argument("-NA", default=15000, type=float, help="Effective population sieze of a population ancestral to population 1 and population 2. Default=1500")
parser.add_argument("-T", default=5000, type=float, help="Split time between population 1 and population 2. Default=2000")
parser.add_argument("-t0", default=500, type=float, help="Time when migrations between population 1 and population 2 stop. Default=500")
parser.add_argument("-m", default=20e-5, type=float, help="Symetric Migration rate. Default=10e-5")
parser.add_argument("--Ti_MSMC", default=False, action="store_true", help="whether use the same time boundaries from MSMC for fitting(recommended). Default=False")
parser.add_argument("-n_T", default=1000, type=int, help="Number of time segments in total in fitting. Default=1000")
parser.add_argument("-N0", default=20000, type=float, help="Average effective population size. Default=20000") 
parser.add_argument("--noMig", default=False, action="store_true", help="Option for estimating the migration rates or not(recommended). Default=False")
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

if args.noMig:
    init_params = np.array([math.log(args.N1), math.log(args.N2), math.log(args.NA), math.log(args.T)])
    Scaled_Params = fmin_powell(MSMC_IM_funcs.scaled_chi_square, init_params, args=(T_i, realTMRCA), xtol=1e-4, ftol=1e-8)
    init_chisquare = MSMC_IM_funcs.scaled_chi_square(init_params, T_i, realTMRCA)
    final_chisquare = MSMC_IM_funcs.scaled_chi_square(Scaled_Params, T_i, realTMRCA)
    print("initial chi square value is", init_chisquare,"Final chi square value is", final_chisquare)
    print("N1", "N2", "NA", "T", sep="\t")
    print(math.exp(Scaled_Params[0]), math.exp(Scaled_Params[1]), math.exp(Scaled_Params[2]), math.exp(Scaled_Params[3]), sep="\t")
else:
    init_params = np.array([math.log(args.N1), math.log(args.N2), math.log(args.NA), math.log(args.T),10000*math.atanh(2e3*args.m-1), math.log(args.t0)])
    init_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0(init_params, T_i, realTMRCA)
    Scaled_Params = fmin_powell(MSMC_IM_funcs.scaled_chi_square_Mstopt0, init_params, args=(T_i, realTMRCA), xtol=1e-4, ftol=1e-8)
    final_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0(Scaled_Params, T_i, realTMRCA)
    m = (math.tanh(Scaled_Params[4]/10000)+1)/2e3
    print("initial chi square value is", init_chisquare,"Final chi square value is", final_chisquare)
    print("N1", "N2", "NA", "T", "m", "t0", sep="\t")
    print(math.exp(Scaled_Params[0]), math.exp(Scaled_Params[1]), math.exp(Scaled_Params[2]), math.exp(Scaled_Params[3]), m, math.exp(Scaled_Params[5]), sep="\t")
