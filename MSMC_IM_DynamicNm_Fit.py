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
parser_dist.add_argument('-m', default=10e-5, type=float, help='Symetric Migration rate. Default=10e-5')
parser.add_argument("-n_T", default=1000, type=int, help="Number of time segments in total in fitting")
parser.add_argument("-N0", default=20000, type=float, help="Average effective population size")
parser_dist.add_argument('--Integral', default=False, action="store_true", help="Option for printing integral over tMRCA distribution. (Optional)")
parser_dist.add_argument('Input', action='append', help='Use output file from MSMC for printing tMRCA from MSMC(Optional)')
parser_chisq = subparsers.add_parser('chisq', parents=[parent_parser], help='Print out the chi-square value by calculating the differenece between the TMRCA distribution from MSMC result and IM-based computation')
parser_chisq.add_argument('Input', help='Output file from MSMC')
parser_opt = subparsers.add_parser('opt', parents=[parent_parser], help='Save the plot of inferred parameters in the same directory where Input is by default')
parser_opt.add_argument('Input', help='Output file from MSMC')
parser_opt.add_argument('-Max_m', default=0.001, type=float, help="Maximum migration rates allowed per generation in the optimazition. Default=0.001,")
parser_opt.add_argument('--noMig', default=False, action="store_true", help="Option of turning off migration completely")
parser_opt.add_argument('--xlog', default=False, action="store_true", help="Plot in log scale on x-axis")
#parser_opt.add_argument('--tmrca_csv', default=False, action="store_true", help="Print out tmrca distribution for sanity check. Default=False")
parser_opt.add_argument('--params_csv', default=False, action="store_true", help="Print out parameters. Default=False")#"Plot the inferred paramerters in pdf file(recommended). Otherwise, print out inferred parameters directly(default)")
parser_opt.add_argument('--lambdas_csv', default=False, action="store_true", help="Print out lambdas calculated from theorectical tMRCA. Default=False")
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
    scale = 1/args.Max_m                  
    time_lr_boundaries, lambdas_00, lambdas_01, lambdas_11 = MSMC_IM_funcs.read_lambdas_from_MSMC(args.Input) #time_lr_boundaries=[[left1,right1], []... []]
    left_boundaries = [k[0] for k in time_lr_boundaries] 
    T_i = np.copy(left_boundaries)
    realTMRCA_00 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_00)
    realTMRCA_01 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_01)
    realTMRCA_11 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, lambdas_11)
    realTMRCA = np.array((realTMRCA_00, realTMRCA_01, realTMRCA_11))
    
    All_Init_chisquare=[]
    All_Final_chisquare=[]
    All_T_prior=[]
    All_best_params=[]
    for T_index in range(len(left_boundaries)):  #len(left_boundaries) is the total number of time intervals
#    if True:
#        T_index=5#37
        if T_index != len(left_boundaries) - 1:
            Tbounds = [left_boundaries[T_index], left_boundaries[T_index+1]]
        else:
            Tbounds = [left_boundaries[T_index], left_boundaries[T_index] * 4]
        T_prior = (Tbounds[0]+Tbounds[1])/2
        t0_prior = (Tbounds[0]+Tbounds[1])/4
        t0_index = bisect.bisect_right(left_boundaries, t0_prior) - 1       
        if args.noMig:                    
            par_list=[[math.log(args.N1)]*(T_index+1), [math.log(args.N2)]*(T_index+1), [math.log(args.NA)]*(len(left_boundaries)-T_index)]
            init_params = [value for sublist in par_list for value in sublist]
            init_chisquare = MSMC_IM_funcs.scaled_chi_square_DynamicN(init_params, T_prior, T_index, left_boundaries, T_i, realTMRCA)
            Scaled_Params = fmin_powell(MSMC_IM_funcs.scaled_chi_square_DynamicN, init_params, args=(T_prior, T_index, left_boundaries, T_i, realTMRCA), xtol=1e-4, ftol=1e-2)
            final_chisquare = MSMC_IM_funcs.scaled_chi_square_DynamicN(Scaled_Params, T_prior, T_index, left_boundaries, T_i, realTMRCA)
            #print(init_params, Scaled_Params, init_chisquare, final_chisquare)        
        else:
            par_list=[[math.log(args.N1)]*(T_index+1), [math.log(args.N2)]*(T_index+1), [math.log(args.NA)]*(len(left_boundaries)-T_index), [args.m]*(T_index-t0_index+1), [math.log(t0_prior)]]
            init_params = [value for sublist in par_list for value in sublist]
            init_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_mlist(init_params, T_prior, T_index, left_boundaries, T_i, realTMRCA, scale)
            Scaled_Params = fmin_powell(MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_mlist, init_params, args=(T_prior, T_index, left_boundaries, T_i, realTMRCA, scale), xtol=1e-4, ftol=1e-2)
            final_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_mlist(Scaled_Params, T_prior, T_index, left_boundaries, T_i, realTMRCA, scale)
        All_T_prior.append(T_prior)
        All_best_params.append(Scaled_Params)
        All_Init_chisquare.append(init_chisquare)
        All_Final_chisquare.append(final_chisquare)
    #print(All_Init_chisquare, All_Final_chisquare, sep="\n")
    #best_T_index=T_index
    #Best_Scaled_Params=Scaled_Params
    best_T_index=All_Final_chisquare.index(min(All_Final_chisquare))
    Best_Scaled_Params=All_best_params[best_T_index]
    
    N1_List = [math.exp(n1) for n1 in Best_Scaled_Params[:best_T_index+1]]
    N2_List = [math.exp(n2) for n2 in Best_Scaled_Params[best_T_index+1: 2*best_T_index+2]]
    NA_List = [math.exp(na) for na in Best_Scaled_Params[2*best_T_index+2: len(left_boundaries)+best_T_index+2]]
    T = (left_boundaries[best_T_index] + left_boundaries[best_T_index+1]) / 2
    if not args.noMig:   
        m_list = [(math.tanh(m/10000)+1)/scale for m in Best_Scaled_Params[len(left_boundaries)+best_T_index+2:-1]]
        t0 = math.exp(Best_Scaled_Params[-1])
        t0_index = bisect.bisect_right(left_boundaries, t0)-1
    else:
        m = 0
        t0 = T
        t0_index = best_T_index

    if args.params_csv:
        if not args.noMig:
            print("T_index:{}".format(best_T_index),
                "N1:{}".format(N1_List),
                "N2:{}".format(N2_List),
                "NA:{}".format(NA_List),
                "m:{}".format(m_list),
                "T:{}".format(T),
                "t0:{}".format(t0), sep="\n")
            print("time_index", "left_time_boundary", "right_time_boundary", "N1", "N2", "NA", "m", sep="\t")
            for i in range(len(left_boundaries)):
                if i < t0_index:
                    print(i, left_boundaries[i], left_boundaries[i+1], N1_List[i], N2_List[i], "NA", "0", sep="\t")
                elif i < best_T_index:
                    print(i, left_boundaries[i], left_boundaries[i+1], N1_List[i], N2_List[i], "NA", m[i-t0_index], sep="\t")
                elif i ==  best_T_index:
                    print(i, left_boundaries[i], T, N1_List[i], N2_List[i], "NA", m[i-t0_index], sep="\t")
                    print(i, T, left_boundaries[i+1], "NA", "NA", NA_List[i-best_T_index], "NA", sep="\t")
                elif i > best_T_index:
                    if not i == len(left_boundaries) - 1:
                        print(i, left_boundaries[i], left_boundaries[i+1], "NA", "NA", NA_List[i-best_T_index], "NA", sep="\t")
                    else:
                        print(i, left_boundaries[i], left_boundaries[i] * 4, "NA", "NA", NA_List[i-best_T_index], "NA", sep="\t")
            if args.lambdas_csv:   
                Lambdas = []
                times = np.copy(left_boundaries)
                Integrals = []
                Integral_errs = []
                Lambdas = []
                computedTMRCA = []
                for x_0 in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]]:
                    Err = []
                    Integral = []
                    P_tMRCA = []
                    Lambda_t = []
                    List_x_vector = MSMC_IM_funcs.makeQpropagator_xvector_mlist(x_0, left_boundaries, N1_List, N2_List, m, t0, T, t0_index, best_T_index)
                    for t in times:    
                        Integ, err = MSMC_IM_funcs.F_computeTMRCA_t0_DynamicN_caltbound(t, List_x_vector, left_boundaries, N1_List, N2_List, NA_List, T, best_T_index)
                        Integral.append(Integ)
                        Err.append(err)
                        P_t = MSMC_IM_funcs.computeTMRCA_t0_DynamicN_caltbound(t, List_x_vector, left_boundaries, N1_List, N2_List, NA_List, T, best_T_index)
                        P_tMRCA.append(P_t)
                        lambda_t = P_t / (1 - Integ)
                        Lambda_t.append(lambda_t)
                    computedTMRCA.append(P_tMRCA)
                    Lambdas.append(Lambda_t)
                    Integrals.append(Integral)
                    Integral_errs.append(Err)
                lambda00, lambda01, lambda11 = Lambdas
                
                Filename='/MSMC_IM_DynamitcFit.'+os.path.basename(args.Input)
                sys.stdout = open(os.path.dirname(args.Input)+Filename+'.allinfor.txt', "w")
                total_chi_square = []
                for realtmrca, computedtmrca in zip(realTMRCA, computedTMRCA):
                    chi_square = sum([(realtmrca[i]-computedtmrca[i])**2/realtmrca[i] for i in range(len(T_i))])
                    total_chi_square.append(chi_square)
                print("Given input parameters, the difference measured by chi-square between IM-modelled tMRCA and MSMC-modelled tMRCA is {}".format(sum(total_chi_square)))
                print("Chi_square_00:{}".format(total_chi_square[0]), "Chi_square_01:{}".format(total_chi_square[1]), "Chi_square_11:{}".format(total_chi_square[2]), "correspondingly")
                print("#######################################")             
                print("left_boundaries", "Integral_IM_tRMCA_00", "Integral_Err_00", "Integral_IM_tRMCA_01", "Integral_Err_01", "Integral_IM_tRMCA_11", "Integral_Err_11", "lambda00", "lambda01", "lambda11", "relativeCCR","IM_tMRCA_00", "IM_tMRCA_01", "IM_tMRCA_11", "MSMC_tMRCA_00", "MSMC_tMRCA_01", "MSMC_tMRCA_11", sep="\t")
                for t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, lambda_00, lambda_01, lambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11 in zip(times, Integrals[0], Integral_errs[0], Integrals[1], Integral_errs[1], Integrals[2], Integral_errs[2], lambda00, lambda01, lambda11, relativeCCR, computedTMRCA[0], computedTMRCA[1], computedTMRCA[2], realTMRCA_00, realTMRCA_01, realTMRCA_11):
                    print(t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, lambda_00, lambda_01, lambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11, sep="\t")
                    
        else:
            print("T_index:{}".format(best_T_index),
                "N1:{}".format(N1_List),
                "N2:{}".format(N2_List),
                "NA:{}".format(NA_List),
                "m:{}".format(m),
                "T:{}".format(T),
                "t0:{}".format(t0), sep="\n")
            print("time_index", "left_time_boundary", "right_time_boundary", "N1", "N2", "NA", sep="\t")
            for i in range(len(left_boundaries)):
                if i < best_T_index:
                    print(i, left_boundaries[i], left_boundaries[i+1], N1_List[i], N2_List[i], "NA",sep="\t")
                elif i ==  best_T_index:
                    print(i, left_boundaries[i], T, N1_List[i], N2_List[i], "NA", sep="\t")
                    print(i, T, left_boundaries[i+1], "NA", "NA", NA_List[i-best_T_index], sep="\t")
                elif i > best_T_index:
                    if not i == len(left_boundaries) - 1:
                        print(i, left_boundaries[i], left_boundaries[i+1], "NA", "NA", NA_List[i-best_T_index], sep="\t")
                    else:
                        print(i, left_boundaries[i], left_boundaries[i] * 4, "NA", "NA", NA_List[i-best_T_index], sep="\t")
            if args.lambdas_csv:   
                Lambdas = []
                times = np.copy(left_boundaries)
                Integrals = []
                Integral_errs = []
                Lambdas = []
                computedTMRCA = []
                for x_0 in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]]:
                    Err = []
                    Integral = []
                    P_tMRCA = []
                    Lambda_t = []
                    List_x_vector = MSMC_IM_funcs.makeQpropagator_xvector_mlist(x_0, left_boundaries, N1_List, N2_List, m, t0, T, t0_index, best_T_index)
                    for t in times:    
                        Integ, err = MSMC_IM_funcs.F_computeTMRCA_t0_DynamicN_caltbound(t, List_x_vector, left_boundaries, N1_List, N2_List, NA_List, T, best_T_index)
                        Integral.append(Integ)
                        Err.append(err)
                        P_t = MSMC_IM_funcs.computeTMRCA_t0_DynamicN_caltbound(t, List_x_vector, left_boundaries, N1_List, N2_List, NA_List, T, best_T_index)
                        P_tMRCA.append(P_t)
                        lambda_t = P_t / (1 - Integ)
                        Lambda_t.append(lambda_t)
                    computedTMRCA.append(P_tMRCA)
                    Lambdas.append(Lambda_t)
                    Integrals.append(Integral)
                    Integral_errs.append(Err)
                lambda00, lambda01, lambda11 = Lambdas
                
                Filename='/MSMC_IM_DynamitcFitNoMig.'+os.path.basename(args.Input)
                sys.stdout = open(os.path.dirname(args.Input)+Filename+'.allinfor.txt', "w")
                total_chi_square = []
                for realtmrca, computedtmrca in zip(realTMRCA, computedTMRCA):
                    chi_square = sum([(realtmrca[i]-computedtmrca[i])**2/realtmrca[i] for i in range(len(T_i))])
                    total_chi_square.append(chi_square)
                print("Given input parameters, the difference measured by chi-square between IM-modelled tMRCA and MSMC-modelled tMRCA is {}".format(sum(total_chi_square)))
                print("Chi_square_00:{}".format(total_chi_square[0]), "Chi_square_01:{}".format(total_chi_square[1]), "Chi_square_11:{}".format(total_chi_square[2]), "correspondingly")
                print("#######################################")             
                print("left_boundaries", "Integral_IM_tRMCA_00", "Integral_Err_00", "Integral_IM_tRMCA_01", "Integral_Err_01", "Integral_IM_tRMCA_11", "Integral_Err_11", "lambda00", "lambda01", "lambda11", "relativeCCR","IM_tMRCA_00", "IM_tMRCA_01", "IM_tMRCA_11", "MSMC_tMRCA_00", "MSMC_tMRCA_01", "MSMC_tMRCA_11", sep="\t")
                for t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, lambda_00, lambda_01, lambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11 in zip(times, Integrals[0], Integral_errs[0], Integrals[1], Integral_errs[1], Integrals[2], Integral_errs[2], lambda00, lambda01, lambda11, relativeCCR, computedTMRCA[0], computedTMRCA[1], computedTMRCA[2], realTMRCA_00, realTMRCA_01, realTMRCA_11):
                    print(t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, lambda_00, lambda_01, lambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11, sep="\t")
                
    else: #OUTPUT FORMAT -- PDF FILE(plots)
        # calculate lambda value for the second and third plot in the pdf file
        Lambdas = []
        times = np.copy(left_boundaries)
#        T_i = [(left_boundaries[i]+left_boundaries[i+1])/2 for i in range(len(left_boundaries)-1)]        
        Integrals = []
        Integral_errs = []
        Lambdas = []
        computedTMRCA = []
        for x_0 in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]]:
            Err = []
            Integral = []
            P_tMRCA = []
            Lambda_t = []
            List_x_vector = MSMC_IM_funcs.makeQpropagator_xvector(x_0, left_boundaries, N1_List, N2_List, m, t0, T, t0_index, best_T_index)
            for t in times:    
                Integ, err = MSMC_IM_funcs.F_computeTMRCA_t0_DynamicN_caltbound(t, List_x_vector, left_boundaries, N1_List, N2_List, NA_List, T, best_T_index)
                Integral.append(Integ)
                Err.append(err)
                P_t = MSMC_IM_funcs.computeTMRCA_t0_DynamicN_caltbound(t, List_x_vector, left_boundaries, N1_List, N2_List, NA_List, T, best_T_index)
                P_tMRCA.append(P_t)
                lambda_t = P_t / (1 - Integ)
                Lambda_t.append(lambda_t)
            computedTMRCA.append(P_tMRCA)
            Lambdas.append(Lambda_t)
            Integrals.append(Integral)
            Integral_errs.append(Err)
        lambda00, lambda01, lambda11 = Lambdas
        relativeCCR = [lambda_01 * 2 / (lambda_00 + lambda_11) for lambda_00, lambda_01, lambda_11 in zip(lambda00, lambda01, lambda11)]
        # computedTMRCA_00 = MSMC_IM_funcs.cal_tmrca_IM([1,0,0,0,0], left_boundaries, N1_List, N2_List, NA_List, m, t0, T, t0_index, best_T_index, times)
        # computedTMRCA_01 = MSMC_IM_funcs.cal_tmrca_IM([0,1,0,0,0], left_boundaries, N1_List, N2_List, NA_List, m, t0, T, t0_index, best_T_index, times)
        # computedTMRCA_11 = MSMC_IM_funcs.cal_tmrca_IM([0,0,1,0,0], left_boundaries, N1_List, N2_List, NA_List, m, t0, T, t0_index, best_T_index, times)
        # computedTMRCA = np.array((computedTMRCA_00, computedTMRCA_01, computedTMRCA_11))
        # for x_0 in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]]:
        #      List_x_vector = MSMC_IM_funcs.makeQpropagator_xvector(x_0, left_boundaries, N1_List, N2_List, m, t0, T, t0_index, best_T_index)
        #      lambda_list = [MSMC_IM_funcs.cal_coalescence_rate_bytmrca(t, List_x_vector, left_boundaries, N1_List, N2_List, NA_List, T, best_T_index) for t in T_i]
        #      Lambdas.append(lambda_list)
        if args.lambdas_csv:   
            for i, lambda_00, lambda_01, lambda_11 in zip(range(len(T_i[1:])), Lambdas[0], Lambdas[1], Lambdas[2]):
                 print(T_i[i], T_i[i+1], lambda_00, lambda_01, lambda_11, sep="\t")
        else:
            if not args.noMig:
                Filename='/MSMC_IM_DynamitcFit.'+os.path.basename(args.Input)
            else:
                Filename='/MSMC_IM_DynamitcFitNoMig.'+os.path.basename(args.Input)
            
            sys.stdout = open(os.path.dirname(args.Input)+Filename+'.allinfor.txt', "w")
            total_chi_square = []
            for realtmrca, computedtmrca in zip(realTMRCA, computedTMRCA):
                chi_square = sum([(realtmrca[i]-computedtmrca[i])**2/realtmrca[i] for i in range(len(T_i))])
                total_chi_square.append(chi_square)
            print("Given input parameters, the difference measured by chi-square between IM-modelled tMRCA and MSMC-modelled tMRCA is {}".format(sum(total_chi_square)))
            print("Chi_square_00:{}".format(total_chi_square[0]), "Chi_square_01:{}".format(total_chi_square[1]), "Chi_square_11:{}".format(total_chi_square[2]), "correspondingly")
            print("#######################################")             
            print("left_boundaries", "Integral_IM_tRMCA_00", "Integral_Err_00", "Integral_IM_tRMCA_01", "Integral_Err_01", "Integral_IM_tRMCA_11", "Integral_Err_11", "lambda00", "lambda01", "lambda11", "relativeCCR","IM_tMRCA_00", "IM_tMRCA_01", "IM_tMRCA_11", "MSMC_tMRCA_00", "MSMC_tMRCA_01", "MSMC_tMRCA_11", sep="\t")
            for t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, lambda_00, lambda_01, lambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11 in zip(times, Integrals[0], Integral_errs[0], Integrals[1], Integral_errs[1], Integrals[2], Integral_errs[2], lambda00, lambda01, lambda11, relativeCCR, computedTMRCA[0], computedTMRCA[1], computedTMRCA[2], realTMRCA_00, realTMRCA_01, realTMRCA_11):
                print(t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, lambda_00, lambda_01, lambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11, sep="\t")
                
            sys.stdout = open(os.path.dirname(args.Input)+Filename+'.parameters.txt', "w")
            print("time_index", "left_time_boundary", "right_time_boundary", "N1", "N2", "NA", sep="\t")#, "Init-Chi-Square", "Final-Chi-Square", sep="\t")
            for i in range(len(left_boundaries)):
                if i < best_T_index:
                    print(i, left_boundaries[i], left_boundaries[i+1], N1_List[i], N2_List[i], "NA", sep="\t")#, All_Init_chisquare[i], All_Final_chisquare[i], sep="\t")
                elif i ==  best_T_index:
                    print(i, left_boundaries[i], T, N1_List[i], N2_List[i], "NA", sep="\t")
                    print(i, T, left_boundaries[i+1], "NA", "NA", NA_List[i-best_T_index], sep="\t")#,  All_Init_chisquare[i], All_Final_chisquare[i], sep="\t")
                elif i > best_T_index:
                    if not i == len(left_boundaries) - 1:
                        print(i, left_boundaries[i], left_boundaries[i+1], "NA", "NA", NA_List[i-best_T_index], sep="\t")#, All_Init_chisquare[i], All_Final_chisquare[i], sep="\t")
                    else:
                        print(i, left_boundaries[i], left_boundaries[i] * 4, "NA", "NA", NA_List[i-best_T_index], sep="\t")
               
            x_t_bef = [t for t in left_boundaries[0:best_T_index+1]]
            x_t_bef.append(T)
            N1_List.append(N1_List[-1])
            N2_List.append(N2_List[-1])
            x_t_aft = [t for t in left_boundaries[best_T_index+1:]]
            x_t_aft.insert(0, T)
            x_t_aft.append(left_boundaries[-1]*4)
            NA_List.append(NA_List[-1])
                     
            plot = plt.semilogx if args.xlog else plt.plot
            plt.figure(figsize=(8,13)) 
            plt.subplot(311)
            if not args.noMig:
                plt.figtext(0.1, 0.9, 'T:%s (generations). delat_t_m:%s (generations)'%(realT, delta_t_m), fontsize=14)
                plt.axvspan(t0, T, facecolor='grey', alpha=0.5)
                plt.axvline(x=realT, color='black', linestyle='dashed')
            else:
                plt.figtext(0.4, 0.9, 'T:%s (generations)'%(T), fontsize=14)
                #plt.figtext(0.2, 0.9, 'T:%s (generations). m:%s (per generation). t0:%s (generations)'%(T, m, t0), fontsize=14)
            plot(x_t_bef, N1_List, label='Population 1', drawstyle='steps-post', c='red')
            plot(x_t_bef, N2_List, label='Population 2',drawstyle='steps-post', c='green')
            plot(x_t_aft, NA_List, label='Ancestral Population', drawstyle='steps-post', c='black')
            plt.legend(prop={'size': 10})
            plt.xlabel("Generations ago", fontsize=14)
            plt.ylabel("Population Sizes", fontsize=14)
            plt.subplot(312)
            plot(times, lambda00, label='Within Population 1', drawstyle='steps-post', c='red')
            plot(times, lambda11, label='Within Population 2', drawstyle='steps-post', c='green')
            plot(times, lambda01, label='Cross Pop1 and Pop2', drawstyle='steps-post', c='blue')
            plt.legend(prop={'size': 10})
            plt.ylim(ymin=0)
            plt.xlabel("Generations ago", fontsize=14)
            plt.ylabel("coalescence rate", fontsize=14)
            plt.subplot(313)
            plot(times, relativeCCR, drawstyle='steps-post', c='black')
            plt.ylim((0,1))
            plt.xlabel("Generations ago", fontsize=14)
            plt.ylabel("relative cross-coalescence rate", fontsize=14)
            plt.savefig(os.path.dirname(args.Input)+Filename+'.pdf')
