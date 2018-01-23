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
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('-N1', default=15000, type=float, help='Effective population size of Pop1 which is constant. Default=15000')
parent_parser.add_argument('-N2', default=15000, type=float, help='Effective population size of Pop2 which is constant. Default=15000')
parent_parser.add_argument('-NA', default=15000, type=float, help='Effective population size of a population ancestral to population 1 and population 2 which is constant. Default=15000')
parent_parser.add_argument('-T', default=2000, type=float, help='Split time between population 1 and population 2. Default=2000')
#parent_parser.add_argument('-delta_t_m', default=20, type=float, help='The time period when the migration kept going on between pops. Default=20')
parent_parser.add_argument('-t0', default=500, type=float, help='Time when migrations between population 1 and population 2 stop. Default=500')
parent_parser.add_argument('-m', default=10e-5, type=float, help='Symetric Migration rate. Default=10e-5')

subparsers = parser.add_subparsers(dest='subcommand', help='sub-command help')
parser_dist = subparsers.add_parser('dist', help='Print out computed TMRCA distribution from Isolation-Migration model with parameters given.Optional: print out TMRCA distribution from MSMC result')
parser_dist.add_argument('--N1_List', type=str, help="Put a list of population sizes seperated by comma.eg.20000,30000,40000,40000")
parser_dist.add_argument('--N2_List', type=str, help="Put a list of population sizes seperated by comma")
parser_dist.add_argument('--NA_List', type=str, help="Put a list of population sizes seperated by comma")
parser_dist.add_argument('-T', default=2000, type=float, help='Split time between population 1 and population 2. Default=2000')
parser_dist.add_argument('-t0', default=500, type=float, help='Time when migrations between population 1 and population 2 stop. Default=500')
parser_dist.add_argument('-m', default=10e-4, type=float, help='Symetric Migration rate. Default=10e-5')
parser.add_argument("-n_T", default=1000, type=int, help="Number of time segments in total in fitting")
parser.add_argument("-N0", default=20000, type=float, help="Average effective population size")
parser_dist.add_argument('--Integral', default=False, action="store_true", help="Option for printing integral over tMRCA distribution. (Optional)")
parser_dist.add_argument('Input', action='append', help='Use output file from MSMC for printing tMRCA from MSMC(Optional)')
parser_chisq = subparsers.add_parser('chisq', parents=[parent_parser], help='Print out the chi-square value by calculating the differenece between the TMRCA distribution from MSMC result and IM-based computation')
parser_chisq.add_argument('Input', help='Output file from MSMC')
parser_opt = subparsers.add_parser('opt', parents=[parent_parser], help='Save the plot of inferred parameters in the same directory where Input is by default')
parser_opt.add_argument('Input', help='Output file from MSMC')
parser_opt.add_argument('-Max_m', default=0.1, type=float, help="Maximum migration rates allowed per generation in the optimazition. Default=0.001,")
parser_opt.add_argument('-beta', default=0.05, type=float, help="Test for the approporiate value as pelty value")
parser_opt.add_argument('--xlog', default=False, action="store_true", help="Plot in log scale on x-axis")
#parser_opt.add_argument('--tmrca_csv', default=False, action="store_true", help="Print out tmrca distribution for sanity check. Default=False")
parser_opt.add_argument('--params_csv', default=False, action="store_true", help="Print out parameters. Default=False")#"Plot the inferred paramerters in pdf file(recommended). Otherwise, print out inferred parameters directly(default)")
args = parser.parse_args() 

if args.subcommand == 'dist':
    time_pattern = [-math.log(1-i/args.n_T)* 2* args.N0 for i in range(args.n_T)]
    T_i = np.copy(time_pattern)
    N1_List = [float(i) for i in args.N1_List.split(',')]
    N2_List = [float(i) for i in args.N2_List.split(',')]
    NA_List = [float(i) for i in args.NA_List.split(',')]
    m = args.m
    t0 = args.t0
    T = args.T
    t0_index = bisect.bisect_right(time_pattern, t0) - 1
    T_index = bisect.bisect_right(time_pattern, T) - 1
    
    computedTMRCA_00 = MSMC_IM_funcs.cal_tmrca_IM([1,0,0,0,0], time_pattern, N1_List, N2_List, NA_List, m, t0, T, t0_index, best_T_index, T_i)
    computedTMRCA_01 = MSMC_IM_funcs.cal_tmrca_IM([0,1,0,0,0], time_pattern, N1_List, N2_List, NA_List, m, t0, T, t0_index, best_T_index, T_i)
    computedTMRCA_11 = MSMC_IM_funcs.cal_tmrca_IM([0,0,1,0,0], time_pattern, N1_List, N2_List, NA_List, m, t0, T, t0_index, best_T_index, T_i)
    computedTMRCA = np.array((computedTMRCA_00, computedTMRCA_01, computedTMRCA_11))
    
    if not args.Integral:
        print("time_index", "left_time_boundary", "right_time_boundary", "IM_tMRCA_00", "IM_tMRCA_01", "IM_tMRCA_11", sep="\t")
        for i in range(len(left_boundaries)):
            if i != len(left_boundaries) - 1:
                print(i, left_boundaries[i], left_boundaries[i+1], computedTMRCA_00[i], computedTMRCA_01[i], computedTMRCA_11[i], sep="\t")
            else:
                print(i, left_boundaries[i], left_boundaries[i] * 4, computedTMRCA_00[i], computedTMRCA_01[i], computedTMRCA_11[i], sep="\t")
            
    else:
        Integrals = []
        Integral_errs = []
        Integral = []
        Err = []
        for x_0 in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]]:
            List_x_vector = MSMC_IM_funcs.makeQpropagator_xvector(x_0, left_boundaries, N1_List, N2_List, m, t0, T, t0_index, T_index)
            for t in T_i:    
                Integ, err = MSMC_IM_funcs.F_computeTMRCA_t0_DynamicN_caltbound(t, List_x_vector, left_boundaries, N1_List, N2_List, NA_List, T, best_T_index)
                Integral.append(Integ)
                Err.append(err)
            Integrals.append(Integral)
            Integral_errs.append(Err)        
        print("t", "Integral_IM_tRMCA_00", "Integral_IM_tRMCA_01", "Integral_IM_tRMCA_11", "Integral_Err_00", "Integral_Err_01", "Integral_Err_11", sep="\t")
        for t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11 in zip(T_i[1:], Integrals[0], Integral_errs[0], Integrals[1], Integral_errs[1], Integrals[2], Integral_errs[2]):
            print(t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, sep="\t")
             
    if False:
        time_lr_boundaries, lambdas_00, lambdas_01, lambdas_11 = MSMC_IM_funcs.read_lambdas_from_MSMC(args.Input) #time_lr_boundaries=[[left1,right1], []... []]
        left_boundaries = [k[0] for k in time_lr_boundaries] 
        T_i = np.copy(left_boundaries)
        realTMRCA_00 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_00)
        realTMRCA_01 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_01)
        realTMRCA_11 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_11)
        realTMRCA = np.array((realTMRCA_00, realTMRCA_01, realTMRCA_11))
    
        print("time_index", "left_time_boundary", "right_time_boundary", "IM_tMRCA_00", "IM_tMRCA_01", "IM_tMRCA_11", "MSMC_tMRCA_00", "MSMC_tMRCA_01", "MSMC_tMRCA_11", sep="\t")
        for i in range(len(left_boundaries)):
            if i != len(left_boundaries) - 1:
                print(i, left_boundaries[i], left_boundaries[i+1], computedTMRCA_00[i], computedTMRCA_01[i], computedTMRCA_11[i], realTMRCA_00[i], realTMRCA_01[i], realTMRCA_11[i], sep="\t")
            else:
                print(i, left_boundaries[i], left_boundaries[i] * 4, computedTMRCA_00[i], computedTMRCA_01[i], computedTMRCA_11[i], realTMRCA_00[i], realTMRCA_01[i], realTMRCA_11[i], sep="\t")    

elif args.subcommand == 'chisq':
    time_lr_boundaries, lambdas_00, lambdas_01, lambdas_11 = MSMC_IM_funcs.read_lambdas_from_MSMC(args.Input) #time_lr_boundaries=[[left1,right1], []... []]
    left_boundaries = [k[0] for k in time_lr_boundaries] 
    T_i = np.copy(left_boundaries)
    realTMRCA_00 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_00)
    realTMRCA_01 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_01)
    realTMRCA_11 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_11)
    realTMRCA = np.array((realTMRCA_00, realTMRCA_01, realTMRCA_11))

    t0_index = bisect.bisect_right(left_boundaries, args.t0) - 1
    T_index = bisect.bisect_right(left_boundaries, args.T) - 1
    computedTMRCA_00 = MSMC_IM_funcs.cal_tmrca_IM([1,0,0,0,0], left_boundaries, [args.N1] * (T_index + 1), [args.N2] * (T_index + 1), [args.NA]*(len(left_boundaries)-T_index), args.m, args.t0, args.T, t0_index, T_index, T_i)
    computedTMRCA_01 = MSMC_IM_funcs.cal_tmrca_IM([0,1,0,0,0], left_boundaries, [args.N1] * (T_index + 1), [args.N2] * (T_index + 1), [args.NA]*(len(left_boundaries)-T_index), args.m, args.t0, args.T, t0_index, T_index, T_i)
    computedTMRCA_11 = MSMC_IM_funcs.cal_tmrca_IM([0,0,1,0,0], left_boundaries, [args.N1] * (T_index + 1), [args.N2] * (T_index + 1), [args.NA]*(len(left_boundaries)-T_index), args.m, args.t0, args.T, t0_index, T_index, T_i)
    computedTMRCA = np.array((computedTMRCA_00, computedTMRCA_01, computedTMRCA_11))
    
    total_chi_square = []
    for realtmrca, computedtmrca in zip(realTMRCA, computedTMRCA):
        chi_square = sum([(realtmrca[i]-computedtmrca[i])**2/realtmrca[i] for i in range(len(T_i))])
        total_chi_square.append(chi_square)
    print("Given input parameters, the difference measured by chi-square between IM-modelled tMRCA and MSMC-modelled tMRCA is {}".format(sum(total_chi_square)))
    print("Chi_square_00:{}".format(total_chi_square[0]), "Chi_square_01:{}".format(total_chi_square[1]), "Chi_square_11:{}".format(total_chi_square[2]), "correspondingly")
    print("time_index", "left_time_boundary", "right_time_boundary", "IM_tMRCA_00", "IM_tMRCA_01", "IM_tMRCA_11", "MSMC_tMRCA_00", "MSMC_tMRCA_01", "MSMC_tMRCA_11", sep="\t")
    for i in range(len(left_boundaries)):
        if i != len(left_boundaries) - 1:
            print(i, left_boundaries[i], left_boundaries[i+1], computedTMRCA_00[i], computedTMRCA_01[i], computedTMRCA_11[i], realTMRCA_00[i], realTMRCA_01[i], realTMRCA_11[i], sep="\t")
        else:
            print(i, left_boundaries[i], left_boundaries[i] * 4, computedTMRCA_00[i], computedTMRCA_01[i], computedTMRCA_11[i], realTMRCA_00[i], realTMRCA_01[i], realTMRCA_11[i], sep="\t")
                     
elif args.subcommand == 'opt':
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
    init_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_mlist(init_params, args.beta, left_boundaries, T_i, realTMRCA, scale)
    Scaled_Params = fmin_powell(MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_mlist, init_params, args=(args.beta, left_boundaries, T_i, realTMRCA, scale), disp=0, xtol=1e-4, ftol=1e-2)
    final_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_mlist(Scaled_Params, args.beta, left_boundaries, T_i, realTMRCA, scale)

    #print(All_Init_chisquare, All_Final_chisquare, sep="\n")
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
               
    print("##############################################################################")          
    print("The final difference measured by chi-square between the theory modeled tMRCA and MSMC modeled tMRCA is {}".format(sum(total_chi_square)))
    print("Chi_square_00:{}".format(total_chi_square[0]), "Chi_square_01:{}".format(total_chi_square[1]), "Chi_square_11:{}".format(total_chi_square[2]), "correspondingly")
    print("##############################################################################")  
    print("left_boundaries", "Integral_IM_tRMCA_00", "Integral_Err_00", "Integral_IM_tRMCA_01", "Integral_Err_01", "Integral_IM_tRMCA_11", "Integral_Err_11", "lambda00", "lambda01", "lambda11", "relativeCCR","IM_tMRCA_00", "IM_tMRCA_01", "IM_tMRCA_11", "MSMC_tMRCA_00", "MSMC_tMRCA_01", "MSMC_tMRCA_11", sep="\t")
    for t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, lambda_00, lambda_01, lambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11 in zip(left_boundaries, Integrals[0], Integral_errs[0], Integrals[1], Integral_errs[1], Integrals[2], Integral_errs[2], lambda00, lambda01, lambda11, relativeCCR, computedTMRCA[0], computedTMRCA[1], computedTMRCA[2], realTMRCA_00, realTMRCA_01, realTMRCA_11):
        print(t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, lambda_00, lambda_01, lambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11, sep="\t")
            
#    else: #OUTPUT FORMAT -- PDF FILE(plots)  
#    Filename='/Plot.sumMsquare.B_{}.xlog.'.format(args.beta)+os.path.basename(args.Input)   
#    Filename='/Plot.sumM.B_{}.xlog.'.format(args.beta)+os.path.basename(args.Input)                     
    Filename='/Plot.Nopenalty.xlog.UniM2.'+os.path.basename(args.Input)        
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
