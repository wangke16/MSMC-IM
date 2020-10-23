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

parser = argparse.ArgumentParser(prog='MSMC_IM', description='Estimate time-dependetn migration rates through fitting IM model to coalescent rates fromMSMC')
parser.add_argument('Input', help='Time dependent within-pop coalescent rates and cross-pop coalescent rates for a pair of populations, e.g. twopops.combined.msmc2.final.txt/twopops.msmc.final.txt')
parser.add_argument('-mu', default=1.25e-8, type=float, help='mutation rate of species. Default is 1.25e-8 for human')
parser.add_argument('-o', help='output directory and prefix of output', action='store')
parser.add_argument('-N1', default=15000, type=float, help='Initial constant effective population size of Pop1 to start fitting process. Default=15000')
parser.add_argument('-N2', default=15000, type=float, help='Initial constatnt effective population size of Pop2 to start fitting process. Default=15000')
parser.add_argument('-m', default=10e-5, type=float, help='Initial symmetric migration rate between two pops to start fitting process. Default=0')
parser.add_argument('-p', default="1*2+25*1+1*2+1*3", type=str, help='Pattern of fixed time segments [default=1*2+25*1+1*2+1*3(MSMC2)], which has to be consistent with MSMC2 or MSMC(default=10*1+15*2) output you are using here')
parser.add_argument('-beta', default="1e-8,1e-6", type=str, help="Regularisation on estimating migration rate and population size. The bigger, the stronger penalty is. Default=1e-8,1e-6")
parser.add_argument('--printfittingdetails', default=False, action="store_true", help="Print detailed infomation during fitting process e.g. estimated split time from M(t) midpoint, initial and final Chi-Square value, estimated coalesent rates from IM model. Defaul=False")
parser.add_argument('--plotfittingdetails', default=False, action="store_true", help="Plot IM estiamtes on m(t), M(t),popsize, coalescent rates, in contrast to MSMC estimates. Default=False")
parser.add_argument('--xlog', default=False, action="store_true", help="Plot all parameters in log scale on x-axis. Default=False. Recommend to add this flag.")
parser.add_argument('--ylog', default=False, action="store_true", help="Plot all parameters in log scale on y-axis. Default=False")
args = parser.parse_args()

if not os.path.dirname(args.o): print("output directory required")
beta=[float(args.beta.split(",")[0]), float(args.beta.split(",")[1])]
time_lr_boundaries, msmc_lambdas00, msmc_lambdas01, msmc_lambdas11 = MSMC_IM_funcs.read_lambdas_from_MSMC(args.Input, args.mu) #time_lr_boundaries=[[left1,right1], []... []]
N1_s = [1/(2*lambda_00) for lambda_00 in msmc_lambdas00]
N2_s = [1/(2*lambda_11) for lambda_11 in msmc_lambdas11]
left_boundaries = [k[0] for k in time_lr_boundaries]
right_boundaries = [k[1] for k in time_lr_boundaries]
right_boundaries[-1] = left_boundaries[-1] * 4
T_i = np.copy(left_boundaries)

repeat = []
segs = []
for seg in args.p.strip().split('+'):
    segs_ = int(seg.split('*')[0]) 
    repeat_ = int(seg.split('*')[1])
    repeat.append(repeat_)
    segs.append(segs_)
len_timesegs = sum([repeat_ * segs_ for repeat_, segs_ in zip(repeat,segs)])
if len_timesegs != len(left_boundaries): raise Exception("Input Error! The time pattern should be consistent with MSMC")
msmc_rCCR = [lambda_01 * 2 / (lambda_00 + lambda_11) for lambda_00, lambda_01, lambda_11 in zip(msmc_lambdas00, msmc_lambdas01, msmc_lambdas11)]
ln = repeat[-1] * segs[-1] + repeat[-2] * segs[-2] #Artifically correct lambdas in the most right time interval(s) to lambda into the second most right time interval
if msmc_lambdas00[-(ln+1)] > 1.5 * min(msmc_lambdas00[-ln:]) or msmc_lambdas00[-(ln+1)] < max(msmc_lambdas00[-ln:])/1.5: msmc_lambdas00[-ln:] = [msmc_lambdas00[-(ln+1)]] * ln
if msmc_lambdas01[-(ln+1)] > 1.5 * min(msmc_lambdas01[-ln:]) or msmc_lambdas01[-(ln+1)] < max(msmc_lambdas01[-ln:])/1.5: msmc_lambdas01[-ln:] = [msmc_lambdas01[-(ln+1)]] * ln
if msmc_lambdas11[-(ln+1)] > 1.5 * min(msmc_lambdas11[-ln:]) or msmc_lambdas11[-(ln+1)] < max(msmc_lambdas11[-ln:])/1.5: msmc_lambdas11[-ln:] =[msmc_lambdas11[-(ln+1)]] * ln

realTMRCA_00 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, msmc_lambdas00)
realTMRCA_01 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, msmc_lambdas01)
realTMRCA_11 = MSMC_IM_funcs.read_tmrcadist_from_MSMC(T_i, left_boundaries, msmc_lambdas11)
realTMRCA = np.array((realTMRCA_00, realTMRCA_01, realTMRCA_11))

length = sum(segs)
par_list=[[math.log(args.N1)]*length, [math.log(args.N2)]*length, [math.log(args.m)] * length]
init_params = [value for sublist in par_list for value in sublist]
init_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_Symmlist(init_params, beta, left_boundaries, T_i, realTMRCA, repeat, segs)
Scaled_Params = fmin_powell(MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_Symmlist, init_params, args=(beta, left_boundaries, T_i, realTMRCA, repeat, segs), disp=0, retall=0, xtol=1e-4, ftol=1e-2)
final_chisquare = MSMC_IM_funcs.scaled_chi_square_Mstopt0_DynamicN_Symmlist(Scaled_Params, beta, left_boundaries, T_i, realTMRCA, repeat, segs)

N1_List = []
N2_List = []
m_List = []
uniqN1 = [math.exp(n1) for n1 in Scaled_Params[:length]]
uniqN2 = [math.exp(n2) for n2 in Scaled_Params[length: 2*length]]
uniqm = [math.exp(m) for m in Scaled_Params[2*length:]]
for i in range(len(segs)):
    N1_List = N1_List + np.repeat(uniqN1[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
    N2_List = N2_List + np.repeat(uniqN2[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
    m_List = m_List + np.repeat(uniqm[sum(segs[0:i]):sum(segs[0:i+1])], repeat[i]).tolist()
CumulativeDF = MSMC_IM_funcs.cumulative_Symmigproportion(right_boundaries, m_List)
m_List_prime = [m_List[i] if CumulativeDF[i] <= 0.999 else 1e-30 for i in range(0, len(CumulativeDF))]
N1_List_prime = [(1-M)*n1 + M*2/(1/n1) for n1, n2, M in zip(N1_List, N2_List, CumulativeDF)]
N2_List_prime = [(1-M)*n2 + M*2/(1/n2) for n1, n2, M in zip(N1_List, N2_List, CumulativeDF)]
# N1_List_prime = [(1-M)*n1 + M*4/(1/n1 + 1/n2) for n1, n2, M in zip(N1_List, N2_List, CumulativeDF)]
# N2_List_prime = [(1-M)*n2 + M*4/(1/n1 + 1/n2) for n1, n2, M in zip(N1_List, N2_List, CumulativeDF)]

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
of = open(os.path.dirname(args.o)+'/{}.b1_{}.b2_{}.MSMC_IM.estimates.txt'.format(os.path.basename(args.o),beta[0],beta[1]),"w")
of.write("left_time_boundary\tim_N1\tim_N2\tm\tM\n")

for i in range(len(left_boundaries)):    
    of.write(str(left_boundaries[i]) +"\t"+ str(N1_List_prime[i]) +"\t"+ str(N2_List_prime[i]) +"\t"+ str(m_List[i]) +"\t"+ str(CumulativeDF[i]) +"\n") 
of.close()
        
if args.printfittingdetails:
    total_chi_square = []
    for realtmrca, computedtmrca in zip(realTMRCA, computedTMRCA):
        chi_square = sum([(realtmrca[i]-computedtmrca[i])**2/realtmrca[i] for i in range(len(T_i))])
        total_chi_square.append(chi_square)
    
    if max(CumulativeDF) >= 0.75:
        xVec = [MSMC_IM_funcs.getCDFintersect(left_boundaries, CumulativeDF, 0.25), MSMC_IM_funcs.getCDFintersect(left_boundaries, CumulativeDF, 0.5), MSMC_IM_funcs.getCDFintersect(left_boundaries, CumulativeDF, 0.75)]
        yVec = [0.25, 0.5, 0.75]
    elif max(CumulativeDF) >= 0.5:
        xVec = [MSMC_IM_funcs.getCDFintersect(left_boundaries, CumulativeDF, 0.25), MSMC_IM_funcs.getCDFintersect(left_boundaries, CumulativeDF, 0.5)]
        yVec = [0.25, 0.5]
    elif max(CumulativeDF) >= 0.25:
        xVec = [MSMC_IM_funcs.getCDFintersect(left_boundaries, CumulativeDF, 0.25)]
        yVec = [0.25]
    f = open(os.path.dirname(args.o)+'/{}.b1_{}.b2_{}.MSMC_IM.fittingdetails.txt'.format(os.path.basename(args.o),beta[0],beta[1]),"w")
    if max(CumulativeDF) >= 0.25:
        f.write("The split time is estimated to be around {} gens (i.e. 0.25,0.5,0.75 quantile)".format(xVec) +"\n") 
    f.write("Initial Chi-Square distance is {} and final Chi-Square distance is {}".format(init_chisquare,final_chisquare) +"\n")
    print("##############################################################################")  
    f.write("left_boundaries\tIM_lambda00\tIM_lambda01\tIM_lambda11\tIM_rCCR\tMSMC_lambda00\tMSMC_lambda01\tMSMC_lambda11\tMSMC_rCCR\tmsmc_N1\tmsmc_N2\tnaive_im_N1\tnaive_im_N2\tim_N1\tim_N2\n") #\tunc_m\n")
#    f.write(str(len(left_boundaries)) +"\t"+ str(len(im_lambdas00)) +"\t"+ str(len(im_lambdas01)) +"\t"+ str(len(im_lambdas11)) +"\t"+ str(len(im_rCCR)) +"\t"+ str(len(msmc_lambdas00)) +"\t"+ str(len(msmc_lambdas01)) +"\t"+ str(len(msmc_lambdas11)) +"\t"+ str(len(msmc_rCCR))+"\t"+ str(len(N1_s))+"\t"+ str(len(N2_s))+"\t"+ str(len(N1_List))+"\t"+ str(len(N2_List))+ "\t"+ str(len(N1_List_prime)) +"\t"+ str(len(N2_List_prime)) +"\t"+ str(len(m_List))+"\n")# +"\t"+ length)
    for t, OUTlambda_00, OUTlambda_01, OUTlambda_11, rCCR,  INlambda_00, INlambda_01, INlambda_11, INrCCR, i in zip(left_boundaries, im_lambdas00, im_lambdas01, im_lambdas11, im_rCCR, msmc_lambdas00, msmc_lambdas01, msmc_lambdas11, msmc_rCCR, list(range(len(m_List)))):
        f.write(str(t) +"\t"+ str(OUTlambda_00) +"\t"+ str(OUTlambda_01) +"\t"+ str(OUTlambda_11) +"\t"+ str(rCCR) +"\t"+ str(INlambda_00) +"\t"+ str(INlambda_01) +"\t"+ str(INlambda_11) +"\t"+ str(INrCCR) +"\t"+ str(N1_s[i]) +"\t"+ str(N2_s[i]) +"\t"+ str(N1_List[i]) +"\t"+ str(N2_List[i]) +"\t"+ str(N1_List_prime[i]) +"\t"+ str(N2_List_prime[i]) +"\n")  #str(m_List[i]) +"\n") 
    f.close() 
#    print("left_boundaries", "Integral_IM_tRMCA_00", "Integral_Err_00", "Integral_IM_tRMCA_01", "Integral_Err_01", "Integral_IM_tRMCA_11", "Integral_Err_11", "IM_lambda00", "IM_lambda01", "IM_lambda11", "im_rCCR","IM_tMRCA_00", "IM_tMRCA_01", "IM_tMRCA_11", "MSMC_lambda00", "MSMC_lambda01", "MSMC_lambda11", "MSMC_tMRCA_00", "MSMC_tMRCA_01", "MSMC_tMRCA_11", sep="\t")
#    for t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, Plambda_00, Plambda_01, Plambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, Mlambda_00, Mlambda_01, Mlambda_11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11 in zip(left_boundaries, Integrals[0], Integral_errs[0], Integrals[1], Integral_errs[1], Integrals[2], Integral_errs[2], lambda00, lambda01, lambda11, im_rCCR, computedTMRCA[0], computedTMRCA[1], computedTMRCA[2], msmc_lambdas00, msmc_lambdas01, msmc_lambdas11, realTMRCA_00, realTMRCA_01, realTMRCA_11):
#        print(t, Integral_00, Integral_err_00, Integral_01, Integral_err_01, Integral_11, Integral_err_11, Plambda_00, Plambda_01, Plambda_11, rCCR, P_tMRCA00, P_tMRCA01, P_tMRCA11, Mlambda_00, Mlambda_01, Mlambda_11, MSMC_tMRCA_00, MSMC_tMRCA_01, MSMC_tMRCA_11, sep="\t")

if args.plotfittingdetails:
    if args.xlog and args.ylog:
        ofp=os.path.dirname(args.o)+'/{}.b1_{}.b2_{}.MSMC_IM.fittingdetails.xylog.pdf'.format(os.path.basename(args.o),beta[0],beta[1])
    elif args.xlog:
        ofp=os.path.dirname(args.o)+'/{}.b1_{}.b2_{}.MSMC_IM.fittingdetails.xlog.pdf'.format(os.path.basename(args.o),beta[0],beta[1])
    elif args.ylog:
        ofp=os.path.dirname(args.o)+'/{}.b1_{}.b2_{}.MSMC_IM.fittingdetails.ylog.pdf'.format(os.path.basename(args.o),beta[0],beta[1])
    else:
        ofp=os.path.dirname(args.o)+'/{}.b1_{}.b2_{}.MSMC_IM.fittingdetails.pdf'.format(os.path.basename(args.o),beta[0],beta[1])
    plot = plt.semilogx if args.xlog else plt.plot
    plot2 = plt.loglog if args.ylog and args.xlog else plt.semilogx if args.xlog else plt.semilogy if args.ylog else plt.plot 
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
    plot(left_boundaries, m_List_prime, drawstyle='steps-post', c='gray', label='Symmetric Migration rate')
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
    plot(left_boundaries, CumulativeDF, label='Infer: CDF', c='orange')
    if max(CumulativeDF) >= 0.25:
        plt.stem(xVec, yVec, linefmt=':', c='orange')
    plt.legend(prop={'size': 8})
    plt.xlim(0, 2e5)
    plt.ylim((0,1))
    plt.xlabel("Generations ago", fontsize=12)
    plt.ylabel("relative cross-coalescence rate", fontsize=12)
    plt.savefig(ofp)
