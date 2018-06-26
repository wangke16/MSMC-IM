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
parser.add_argument('-N1', default=15000, type=float, help='Initial effective population size of Pop1. Default=15000')
parser.add_argument('-N2', default=15000, type=float, help='Initial effective population size of Pop2. Default=15000')
parser.add_argument('-m', default=10e-5, type=float, help='Initial symmetric migration rate between two pops. Default=10e-5')
parser.add_argument('-p', default="1*2+25*1+1*2+1*3", type=str, help='Pattern of fixed time segments [default=1*2+25*1+1*2+1*3(MSMC2)], which has to be consistent with MSMC2 or MSMC(default=10*1+15*2) output')
parser.add_argument('--beta', nargs='+', help="Values of penaly on migration rate regularisation and pop size regularisation. Default=2.5e-9,1e-5")
parser.add_argument('--run2m', default=False, action="store_true", help="Run the model with two series of migration rates")
parser.add_argument('--xlog', default=False, action="store_true", help="Plot all parameters (expect pop size) in log scale on x-axis. Default=False")
parser.add_argument('--xylog', default=False, action="store_true", help="Plot in log scale on both axis on population sizes. Default=False")
parser.add_argument('--printX2', default=False, action="store_true", help="Print initial and final Chi-Square value. Default=False")
parser.add_argument('--printLambdas', default=False, action="store_true", help="Print lambdas from IM model and from MSMC. Default=False")

#parser.add_argument('--params_csv', default=False, action="store_true", help="Print out parameters. Default=False")#"Plot the inferred paramerters in pdf file(recommended). Otherwise, print out inferred parameters directly(default)")
args = parser.parse_args() 

beta=[float(args.beta[0].split(",")[0]), float(args.beta[0].split(",")[1])]
time_lr_boundaries, msmc_lambdas00, msmc_lambdas01, msmc_lambdas11 = MSMC_IM_funcs.read_lambdas_from_MSMC(args.Input) #time_lr_boundaries=[[left1,right1], []... []]
msmc_rCCR = [lambda_01 * 2 / (lambda_00 + lambda_11) for lambda_00, lambda_01, lambda_11 in zip(msmc_lambdas00, msmc_lambdas01, msmc_lambdas11)]
N1_s = [1/(2*lambda_00) for lambda_00 in msmc_lambdas00]
N2_s = [1/(2*lambda_11) for lambda_11 in msmc_lambdas11]
left_boundaries = [k[0] for k in time_lr_boundaries] 
right_boundaries = [k[1] for k in time_lr_boundaries] 
right_boundaries[-1] = left_boundaries[-1] * 4
T_i = np.copy(left_boundaries)
realTMRCA_00 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, msmc_lambdas00)
realTMRCA_01 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, msmc_lambdas01)
realTMRCA_11 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, msmc_lambdas11)
realTMRCA = np.array((realTMRCA_00, realTMRCA_01, realTMRCA_11))

scale = 1
repeat = []
segs = []
for seg in args.p.strip().split('+'):
    segs_ = int(seg.split('*')[0]) 
    repeat_ = int(seg.split('*')[1])
    repeat.append(repeat_)
    segs.append(segs_)
len_timesegs = sum([repeat_ * segs_ for repeat_, segs_ in zip(repeat,segs)])
if len_timesegs != len(left_boundaries): raise Exception("Input Error! The time pattern should be consistent with MSMC")
length = sum(segs)
if not args.run2m:
    par_list=[[math.log(args.N1)]*length, [math.log(args.N2)]*length, [math.atanh(args.m*2*scale-1)]*length]
    init_params = [value for sublist in par_list for value in sublist]
    init_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_Symmlist(init_params, beta, left_boundaries, T_i, realTMRCA, scale, repeat, segs)
    Scaled_Params = fmin_powell(MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_Symmlist, init_params, args=(beta, left_boundaries, T_i, realTMRCA, scale, repeat, segs), disp=0, retall=0, xtol=1e-4, ftol=1e-2)
    final_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_Symmlist(Scaled_Params, beta, left_boundaries, T_i, realTMRCA, scale, repeat, segs)
    
    N1_List = []
    N2_List = []
    m_List = []
    uniqN1 = [math.exp(n1) for n1 in Scaled_Params[:length]]
    uniqN2 = [math.exp(n2) for n2 in Scaled_Params[length: 2*length]]
    uniqm = [(math.tanh(m)+1)/(2*scale) for m in Scaled_Params[2*length:]]
    for i in range(len(segs)):
        N1_List = N1_List + np.repeat(uniqN1[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
        N2_List = N2_List + np.repeat(uniqN2[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
        m_List = m_List + np.repeat(uniqm[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
    CumulativeDF = MSMC_IM_funcs.cumulative_Symmigproportion(right_boundaries, m_List)
    N1_List_prime = [(1-M)*n1 + M*4/(1/n1 + 1/n2) for n1, n2, M in zip(N1_List, N2_List, CumulativeDF)]
    N2_List_prime = [(1-M)*n2 + M*4/(1/n1 + 1/n2) for n1, n2, M in zip(N1_List, N2_List, CumulativeDF)]
    
    
    Integrals = []
    Integral_errs = []
    Lambdas = []
    computedTMRCA = []
    for x_0 in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]]:
        Err = []
        Integral = [] 
        P_tMRCA = []
        Lambda_t = []
        List_x_vector = MSMC_IM_funcs.makeQpropagator_xvector_Symmlist(x_0, left_boundaries, N1_List, N2_List, m_List)
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
    im_lambdas00, im_lambdas01, im_lambdas11 = Lambdas
    im_rCCR = [lambda_01 * 2 / (lambda_00 + lambda_11) for lambda_00, lambda_01, lambda_11 in zip(im_lambdas00, im_lambdas01, im_lambdas11)]
    print("time_index", "left_time_boundary", "right_time_boundary", "msmc_N1", "msmc_N2", "naive_im_N1", "naive_im_N2", "im_N1", "im_N2", "m", "M", sep="\t")
    for i in range(len(left_boundaries)):    
        if not i == len(left_boundaries) - 1:
            print(i, left_boundaries[i], left_boundaries[i+1], N1_s[i], N2_s[i], N1_List[i], N2_List[i], N1_List_prime[i], N2_List_prime[i], m_List[i], CumulativeDF[i], sep="\t") 
        else:
            print(i, left_boundaries[i], left_boundaries[i] * 4, N1_s[i], N2_s[i], N1_List[i], N2_List[i], N1_List_prime[i], N2_List_prime[i], m_List[i], CumulativeDF[i], sep="\t")
else:
    par_list=[[math.log(args.N1)]*length, [math.log(args.N2)]*length, [math.atanh(args.m*2*scale-1)]*length, [math.atanh(args.m*2*scale-1)]*length]
    init_params = [value for sublist in par_list for value in sublist]
    init_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_2mlist(init_params, beta, left_boundaries, T_i, realTMRCA, scale, repeat, segs)
    Scaled_Params = fmin_powell(MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_2mlist, init_params, args=(beta, left_boundaries, T_i, realTMRCA, scale, repeat, segs), disp=0, retall=0, xtol=1e-4, ftol=1e-2)
    final_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_2mlist(Scaled_Params, beta, left_boundaries, T_i, realTMRCA, scale, repeat, segs)

    N1_List = []
    N2_List = []
    m1_List = []
    m2_List = []
    uniqN1 = [math.exp(n1) for n1 in Scaled_Params[:length]]
    uniqN2 = [math.exp(n2) for n2 in Scaled_Params[length: 2*length]]
    uniqm1 = [(math.tanh(m)+1)/(2*scale) for m in Scaled_Params[2*length:3*length]]
    uniqm2 = [(math.tanh(m)+1)/(2*scale) for m in Scaled_Params[3*length:]]
    for i in range(len(segs)):
        N1_List = N1_List + np.repeat(uniqN1[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
        N2_List = N2_List + np.repeat(uniqN2[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
        m1_List = m1_List + np.repeat(uniqm1[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
        m2_List = m2_List + np.repeat(uniqm2[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
    CumulativeDF = MSMC_IM_funcs.cumulative_2migproportion(right_boundaries, m1_List, m2_List)
    N1_List_prime = [(1-M)*n1 + M*2/(1/n1) for n1, n2, M in zip(N1_List, N2_List, CumulativeDF)]
    N2_List_prime = [(1-M)*n2 + M*2/(1/n2) for n1, n2, M in zip(N1_List, N2_List, CumulativeDF)]
#    N1_List_prime = [(1-M)*n1 + M*4/(1/n1 + 1/n2) for n1, n2, M in zip(N1_List, N2_List, CumulativeDF)]
#    N2_List_prime = [(1-M)*n2 + M*4/(1/n1 + 1/n2) for n1, n2, M in zip(N1_List, N2_List, CumulativeDF)]
    
    Integrals = []
    Integral_errs = []
    Lambdas = []
    computedTMRCA = []
    for x_0 in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]]:
        Err = []
        Integral = [] 
        P_tMRCA = []
        Lambda_t = []
        List_x_vector = MSMC_IM_funcs.makeQpropagator_xvector_2mlist(x_0, left_boundaries, N1_List, N2_List, m1_List, m2_List)
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
    im_lambdas00, im_lambdas01, im_lambdas11 = Lambdas
    im_rCCR = [lambda_01 * 2 / (lambda_00 + lambda_11) for lambda_00, lambda_01, lambda_11 in zip(im_lambdas00, im_lambdas01, im_lambdas11)]
    print("time_index", "left_time_boundary", "right_time_boundary", "N1", "N2", "m1", "m2",sep="\t")
    for i in range(len(left_boundaries)):    
        if not i == len(left_boundaries) - 1:
            print(i, left_boundaries[i], left_boundaries[i+1], N1_List[i], N2_List[i], m1_List[i], m2_List[i], sep="\t")  
        else:
            print(i, left_boundaries[i], left_boundaries[i] * 4, N1_List[i], N2_List[i], m1_List[i], m2_List[i], sep="\t")    
    
total_chi_square = []
for realtmrca, computedtmrca in zip(realTMRCA, computedTMRCA):
    chi_square = sum([(realtmrca[i]-computedtmrca[i])**2/realtmrca[i] for i in range(len(T_i))])
    total_chi_square.append(chi_square)
    
mid_t = [(left_time_boundary + right_time_boundary)/2 for left_time_boundary, right_time_boundary in zip(left_boundaries, right_boundaries)]
if max(CumulativeDF) >= 0.75:
    xVec = [MSMC_IM_funcs.getCDFintersect(left_boundaries, right_boundaries, CumulativeDF, 0.25), MSMC_IM_funcs.getCDFintersect(left_boundaries, right_boundaries, CumulativeDF, 0.5), MSMC_IM_funcs.getCDFintersect(left_boundaries, right_boundaries, CumulativeDF, 0.75)]
    yVec = [0.25, 0.5, 0.75]
elif max(CumulativeDF) >= 0.5:
    xVec = [MSMC_IM_funcs.getCDFintersect(left_boundaries, right_boundaries, CumulativeDF, 0.25), MSMC_IM_funcs.getCDFintersect(left_boundaries, right_boundaries, CumulativeDF, 0.5)]
    yVec = [0.25, 0.5]
elif max(CumulativeDF) >= 0.25:
    xVec = [MSMC_IM_funcs.getCDFintersect(left_boundaries, right_boundaries, CumulativeDF, 0.25)]
    yVec = [0.25]

if max(CumulativeDF) >= 0.25:
    print("##############################################################################")  
    print("The split time is estimated to be around {} gens.".format(xVec)) 
 
if args.printX2:    
    print("##############################################################################")        
    print("Initial Chi Square distance:{}".format(init_chisquare), "Final Chi Square distance:{}".format(final_chisquare), sep="\n")
    
if args.printLambdas:
    print("##############################################################################")  
    print("left_boundaries", "IM_lambda00", "IM_lambda01", "IM_lambda11", "IM_rCCR", "MSMC_lambda00", "MSMC_lambda01", "MSMC_lambda11", "MSMC_rCCR", sep="\t")
    for t, OUTlambda_00, OUTlambda_01, OUTlambda_11, rCCR,  INlambda_00, INlambda_01, INlambda_11, INrCCR in zip(left_boundaries, im_lambdas00, im_lambdas01, im_lambdas11, im_rCCR, msmc_lambdas00, msmc_lambdas01, msmc_lambdas11, msmc_rCCR):
        print(t, OUTlambda_00, OUTlambda_01, OUTlambda_11, rCCR, INlambda_00, INlambda_01, INlambda_11, INrCCR, sep="\t") 
#    print("left_boundaries", "Integral_IM_tRMCA_00", "Integral_Err_00", "Integral_IM_tRMCA_01", "Integral_Err_01", "Integral_IM_tRMCA_11", "Integral_Err_11", "IM_lambda00", "IM_lambda01", "IM_lambda11", "im_rCCR","IM_tMRCA_00", "IM_tMRCA_01", "IM_tMRCA_11", "MSMC_lambda00", "MSMC_lambda01", "MSMC_lambda11", "MSMC_tMRCA_00", "MSMC_tMRCA_01", "MSMC_tMRCA_11", sep="\t")
#    for t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, Plambda_00, Plambda_01, Plambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, Mlambda_00, Mlambda_01, Mlambda_11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11 in zip(left_boundaries, Integrals[0], Integral_errs[0], Integrals[1], Integral_errs[1], Integrals[2], Integral_errs[2], lambda00, lambda01, lambda11, im_rCCR, computedTMRCA[0], computedTMRCA[1], computedTMRCA[2], msmc_lambdas00, msmc_lambdas01, msmc_lambdas11, realTMRCA_00, realTMRCA_01, realTMRCA_11):
#        print(t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, Plambda_00, Plambda_01, Plambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, Mlambda_00, Mlambda_01, Mlambda_11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11, sep="\t")

if args.run2m:
    if args.xlog and args.xylog:
        Filename='/Plot.chisquare.xylog.twoM.b1{}.b2{}.timepattern.'.format(beta[0],beta[1])+os.path.basename(args.Input)
    elif args.xlog:
        Filename='/Plot.chisquare.xlog.twoM.b1{}.b2{}.timepattern.'.format(beta[0],beta[1])+os.path.basename(args.Input) 
    else:
        Filename='/Plot.chisquare.twoM.b1{}.b2{}.timepattern.'.format(beta[0],beta[1])+os.path.basename(args.Input) 
else:
    if args.xlog and args.xylog:
        Filename='/Plot.chisquare.xylog.SymmetricM.b1{}.b2{}.timepattern.'.format(beta[0],beta[1])+os.path.basename(args.Input)
    elif args.xlog:
        Filename='/Plot.chisquare.xlog.SymmetricM.b1{}.b2{}.absb2.ncor.'.format(beta[0],beta[1])+os.path.basename(args.Input) 
    else:
        Filename='/Plot.chisquare.SymmetricM.b1{}.b2{}.timepattern.'.format(beta[0],beta[1])+os.path.basename(args.Input)      
plot = plt.semilogx if args.xlog else plt.plot
plot2 = plt.loglog if args.xylog else plt.semilogx if args.xlog else plt.plot
plt.figure(figsize=(8,16)) 
plt.subplot(411)
plot2(left_boundaries, N1_List_prime, '--', label='CorrectedInfer:Population 1', drawstyle='steps-post', c='red')
plot2(left_boundaries, N2_List_prime, '--', label='CorrectedInfer:Population 2',drawstyle='steps-post', c='green')
#    plot2(left_boundaries, N1_List, ':', label='Infer:Population 1', drawstyle='steps-post', c='red')
#    plot2(left_boundaries, N2_List, ':', label='Infer:Population 2',drawstyle='steps-post', c='green')
plot2(left_boundaries, N1_s, label='MSMC:Population 1', drawstyle='steps-post', c='red')
plot2(left_boundaries, N2_s, label='MSMC:Population 2',drawstyle='steps-post', c='green')
plt.xlim(0, 2e5)
plt.legend(prop={'size': 8})
plt.xlabel("Generations ago", fontsize=12)
plt.ylabel("Population Sizes", fontsize=12)
plt.subplot(412)
if not args.run2m:
    plot(left_boundaries, m_List, drawstyle='steps-post', c='gray', label='Symmetric Migration rate')
else:
    plot(left_boundaries, m1_List, drawstyle='steps-post', c='gray', label='m1: migration rate moving from pop2 to pop1 forward in time')
    plot(left_boundaries, m2_List, drawstyle='steps-post', c='violet', label='m2: migration rate moving from pop1 to pop2 forward in time')
plt.xlim(0, 2e5)
plt.xlabel("Generations ago", fontsize=12)
plt.ylabel("Migration rates", fontsize=12)
plt.legend(prop={'size': 8})
plt.subplot(413)
plot(left_boundaries, im_lambdas00, '--', label='HazardFunc:Within Pop 1', drawstyle='steps-post', c='red')
plot(left_boundaries, im_lambdas11, '--', label='HazardFunc:Within Pop 2', drawstyle='steps-post', c='green')
plot(left_boundaries, im_lambdas01, '--', label='HazardFunc:Cross Pops', drawstyle='steps-post', c='blue')
plot(left_boundaries, msmc_lambdas00, linewidth=0.2, label='MSMC:Within Pop 1', drawstyle='steps-post', c='red')
plot(left_boundaries, msmc_lambdas11, linewidth=0.2, label='MSMC:Within Pop 2', drawstyle='steps-post', c='green')
plot(left_boundaries, msmc_lambdas01, linewidth=0.2, label='MSMC:Cross Pops', drawstyle='steps-post', c='blue')
plt.legend(prop={'size': 8})
plt.xlim(0, 2e5)
plt.ylim(ymin=0)
plt.xlabel("Generations ago", fontsize=12)
plt.ylabel("coalescence rate", fontsize=12)
plt.subplot(414)
plot(left_boundaries, im_rCCR, '--', label='Infer:rCCR',drawstyle='steps-post', c='black')
plot(left_boundaries, msmc_rCCR, '-.', linewidth=0.3, label='MSMC:rCCR', drawstyle='steps-post', c='black')
plot(mid_t, CumulativeDF, label='Infer: CDF', c='orange')
if max(CumulativeDF) >= 0.25:
    plt.stem(xVec, yVec, linefmt=':', c='orange')
# plot([xVec[0], xVec[0]], [0, yVec[0]], linewidth=0.3, color="pink", linestyle=':')
# plot([1, xVec[0]], [yVec[0], yVec[0]], linewidth=0.3, color="pink", linestyle=':')
# plot([xVec[1], xVec[1]], [0, yVec[1]], linewidth=0.3, color="pink", linestyle=':')
# plot([1, xVec[1]], [yVec[1], yVec[1]], linewidth=0.3, color="pink", linestyle=':')
# plot([xVec[2], xVec[2]], [0, yVec[2]], linewidth=0.3, color="pink", linestyle=':')
# plot([1, xVec[2]], [yVec[2], yVec[2]], linewidth=0.3, color="pink", linestyle=':')
plt.legend(prop={'size': 8})
plt.xlim(0, 2e5)
plt.ylim((0,1))
plt.xlabel("Generations ago", fontsize=12)
plt.ylabel("relative cross-coalescence rate", fontsize=12)
plt.savefig(os.path.dirname(args.Input)+Filename+'.pdf')
