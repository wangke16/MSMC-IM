#!/usr/bin/env python3

import MSMC_IM_funcs
import argparse
import math
import bisect
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fmin_powell
import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process

parser = argparse.ArgumentParser(prog='MSMC_IM_Dynamic', description='Find parameters with the max likelihood for fitting IM model to MSMC')
parser.add_argument('Input', help='Output file from MSMC')
parser.add_argument('-N1', default=15000, type=float, help='Effective population size of Pop1 which is constant. Default=15000')
parser.add_argument('-N2', default=15000, type=float, help='Effective population size of Pop2 which is constant. Default=15000')
parser.add_argument('-m', default=10e-5, type=float, help='Symetric Migration rate. Default=10e-5')
parser.add_argument('-Max_m', default=0.1, type=float, help="Maximum migration rates allowed per generation in the optimazition. Default=0.1,")
parser.add_argument('--xlog', default=False, action="store_true", help="Plot in log scale on x-axis")
parser.add_argument('--params_csv', default=False, action="store_true", help="Print out parameters. Default=False")#"Plot the inferred paramerters in pdf file(recommended). Otherwise, print out inferred parameters directly(default)")
args = parser.parse_args() 

time_lr_boundaries, lambdas_00, lambdas_01, lambdas_11 = MSMC_IM_funcs.read_lambdas_from_MSMC(args.Input) #time_lr_boundaries=[[left1,right1], []... []]
RelativeCCR = [lambda_01 * 2 / (lambda_00 + lambda_11) for lambda_00, lambda_01, lambda_11 in zip(lambdas_00, lambdas_01, lambdas_11)]
N1_s = [1/(2*lambda_00) for lambda_00 in lambdas_00]
N2_s = [1/(2*lambda_11) for lambda_11 in lambdas_11]
left_boundaries = [k[0] for k in time_lr_boundaries] 
T_i = np.copy(left_boundaries)
realTMRCA_00 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_00)
realTMRCA_01 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_01)
realTMRCA_11 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_11)
realTMRCA = np.array((realTMRCA_00, realTMRCA_01, realTMRCA_11))

scale = 1/args.Max_m 
length=len(left_boundaries)
par_list=[[math.log(args.N1)]*length, [math.log(args.N2)]*length, [math.atanh(args.m*2*scale-1)]*length]
init_params = [value for sublist in par_list for value in sublist]
init_KL = MSMC_IM_funcs.Kullback_Leibler(init_params, left_boundaries, lambdas_00, lambdas_01, lambdas_11, scale)

# def callbackF(Params,left_boundaries, lambdas_00, lambdas_01, lambdas_11, scale):
#     global NrIter
#     print(NrIter, MSMC_IM_funcs.Kullback_Leibler(Params,left_boundaries, lambdas_00, lambdas_01, lambdas_11, scale), Params[:length], Params[length: 2*length], Params[2*length:], sep="\t")
#     NrIter += 1
#[xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = fmin_powell(MSMC_IM_funcs.Kullback_Leibler, init_params, callback=callbackF, args=(left_boundaries, lambdas_00, lambdas_01, lambdas_11, scale), disp=1, full_output=1, retall=0, xtol=1e-4, ftol=1e-2)

Scaled_Params = fmin_powell(MSMC_IM_funcs.Kullback_Leibler, init_params, args=(left_boundaries, lambdas_00, lambdas_01, lambdas_11, scale), disp=1, retall=0, xtol=1e-4, ftol=1e-2)
final_KL = MSMC_IM_funcs.Kullback_Leibler(Scaled_Params, left_boundaries, lambdas_00, lambdas_01, lambdas_11, scale)
print("Initial Kullback Leibler distance:{}".format(init_KL), "Final Kullback Leibler distance:{}".format(final_KL), sep="\n")
N1_List = [math.exp(n1) for n1 in Scaled_Params[:length]]
N2_List = [math.exp(n2) for n2 in Scaled_Params[length: 2*length]]
m_list = [(math.tanh(m)+1)/(2*scale) for m in Scaled_Params[2*length:]]

Integrals = []
Integral_errs = []
Lambdas = []
computedTMRCA = []
for x_0 in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]]:
    Err = []
    Integral = [] 
    P_tMRCA = []
    Lambda_t = []
    List_x_vector = MSMC_IM_funcs.makeQpropagator_xvector_mlist(x_0, left_boundaries, N1_List, N2_List, m_list)
    for t in left_boundaries:    
        Integ, err = MSMC_IM_funcs.F_computeTMRCA_t0_DynamicN_caltbound_mlist(t, List_x_vector, left_boundaries, N1_List, N2_List)
        Integral.append(Integ)
        Err.append(err)
        P_t = MSMC_IM_funcs.computeTMRCA_t0_DynamicN_caltbound_mlist(t, List_x_vector, left_boundaries, N1_List, N2_List)
        P_tMRCA.append(P_t)
        lambda_t = P_t / (1 - Integ)
        Lambda_t.append(lambda_t)
    computedTMRCA.append(P_tMRCA)
    Lambdas.append(Lambda_t)
    Integrals.append(Integral)
    Integral_errs.append(Err)
lambda00, lambda01, lambda11 = Lambdas
relativeCCR = [lambda_01 * 2 / (lambda_00 + lambda_11) for lambda_00, lambda_01, lambda_11 in zip(lambda00, lambda01, lambda11)]

total_chi_square = []
for realtmrca, computedtmrca in zip(realTMRCA, computedTMRCA):
    chi_square = sum([(realtmrca[i]-computedtmrca[i])**2/realtmrca[i] for i in range(len(T_i))])
    total_chi_square.append(chi_square)
    
    #    if args.params_csv:
print("time_index", "left_time_boundary", "right_time_boundary", "N1", "N2", "m", sep="\t")
for i in range(len(left_boundaries)):    
    if not i == len(left_boundaries) - 1:
        print(i, left_boundaries[i], left_boundaries[i+1], N1_List[i], N2_List[i], m_list[i], sep="\t")  
    else:
        print(i, left_boundaries[i], left_boundaries[i] * 4, N1_List[i], N2_List[i], m_list[i], sep="\t")
 
if True:           
    print("##############################################################################")          
    print("Initial Kullback Leibler distance:{}".format(init_KL), "Final Kullback Leibler distance:{}".format(final_KL), sep="\n")
    print("##############################################################################")  
    print("left_boundaries", "Integral_IM_tRMCA_00", "Integral_Err_00", "Integral_IM_tRMCA_01", "Integral_Err_01", "Integral_IM_tRMCA_11", "Integral_Err_11", "lambda00", "lambda01", "lambda11", "relativeCCR","IM_tMRCA_00", "IM_tMRCA_01", "IM_tMRCA_11", "MSMC_tMRCA_00", "MSMC_tMRCA_01", "MSMC_tMRCA_11", sep="\t")
    for t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, lambda_00, lambda_01, lambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11 in zip(left_boundaries, Integrals[0], Integral_errs[0], Integrals[1], Integral_errs[1], Integrals[2], Integral_errs[2], lambda00, lambda01, lambda11, relativeCCR, computedTMRCA[0], computedTMRCA[1], computedTMRCA[2], realTMRCA_00, realTMRCA_01, realTMRCA_11):
        print(t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, lambda_00, lambda_01, lambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11, sep="\t")
    #    else: #OUTPUT FORMAT -- PDF FILE(plots)  
    Filename='/Plot.KL.xlog.SymmetricM.'+os.path.basename(args.Input)        
    plot = plt.semilogx if args.xlog else plt.plot
    plt.figure(figsize=(8,16)) 
    plt.subplot(411)
    plot(left_boundaries, N1_List, '--', label='Infer:Population 1', drawstyle='steps-post', c='red')
    plot(left_boundaries, N2_List, '--', label='Infer:Population 2',drawstyle='steps-post', c='green')
    plot(left_boundaries, N1_s, label='MSMC:Population 1', drawstyle='steps-post', c='red')
    plot(left_boundaries, N2_s, label='MSMC:Population 2',drawstyle='steps-post', c='green')
    plt.legend(prop={'size': 8})
    plt.xlabel("Generations ago", fontsize=12)
    plt.ylabel("Population Sizes", fontsize=12)
    plt.subplot(412)
    plot(left_boundaries, m_list, drawstyle='steps-post', c='black')
    plt.xlabel("Generations ago", fontsize=12)
    plt.ylabel("Migration rates", fontsize=12)
    plt.subplot(413)
    plot(left_boundaries, lambda00, '--', label='HazardFunc:Within Pop 1', drawstyle='steps-post', c='red')
    plot(left_boundaries, lambda11, '--', label='HazardFunc:Within Pop 2', drawstyle='steps-post', c='green')
    plot(left_boundaries, lambda01, '--', label='HazardFunc:Cross Pops', drawstyle='steps-post', c='blue')
    plot(left_boundaries, lambdas_00, linewidth=0.2, label='MSMC:Within Pop 1', drawstyle='steps-post', c='red')
    plot(left_boundaries, lambdas_11, linewidth=0.2, label='MSMC:Within Pop 2', drawstyle='steps-post', c='green')
    plot(left_boundaries, lambdas_01, linewidth=0.2, label='MSMC:Cross Pops', drawstyle='steps-post', c='blue')
    plt.legend(prop={'size': 8})
    plt.ylim(ymin=0)
    plt.xlabel("Generations ago", fontsize=12)
    plt.ylabel("coalescence rate", fontsize=12)
    plt.subplot(414)
    plot(left_boundaries, relativeCCR, '--', label='Infer:rCCR',drawstyle='steps-post', c='black')
    plot(left_boundaries, RelativeCCR, linewidth=0.2, label='MSMC:rCCR', drawstyle='steps-post', c='black')
    plt.legend(prop={'size': 8})
    plt.ylim((0,1))
    plt.xlabel("Generations ago", fontsize=12)
    plt.ylabel("relative cross-coalescence rate", fontsize=12)
    plt.savefig(os.path.dirname(args.Input)+Filename+'.pdf')
