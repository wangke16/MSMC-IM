#!/usr/bin/env python3
import math
import bisect
import scipy as sp
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

##Vectors need to be read in: n, tmax, time_boundaries, m, percentile
parser = argparse.ArgumentParser(prog='MSMC_IM results plotter', description='Help on plotting M(t) given time grid, the cumulative probability funtion on m(t)')
parser.add_argument('Input', help='Read in MSMC-IM output [time_index, left_boundaries, right_boundaries, IM_N1, IM_N2, m(t), M(t)]')
parser.add_argument('-o', help='output directory and prefix of output pdf plot', action='store')
parser.add_argument('-tmax', default = 30000, type=float, help="The max time point when evaluating M(t), in generations. Required if no --samegrid flag")
parser.add_argument('-n', default = 100, type=float, help="Width of time intervals between while plotting M(t) between 0 and tmax, in generations. Required if no --samegrid flag")
parser.add_argument('--samegrid', default=False, action="store_true", help="Use the time boundaries from MSMC-IM output for evaluating M(t). Default=False. When turned on, flags -tmax and -n are not required")
args = parser.parse_args() 

def read_from_MSMC_IM(fn):
    time_boundaries = []
    IM_N1s = []
    IM_N2s = []
    m_t_s = []
    f = open(fn, "rt")
    next(f)
    for line in f:
        fields = line.strip().split("\t")
        tLeft = float(fields[0])
        IM_N1 = float(fields[1])
        IM_N2 = float(fields[2])
        m_t = float(fields[3])
        time_boundaries.append(tLeft)
        IM_N1s.append(IM_N1)
        IM_N2s.append(IM_N2)
        m_t_s.append(m_t)
    if time_boundaries[0] or time_boundaries[-1] != float('inf'):
        print("Warning! The time segment is not start from 0 or end with infinity. The program will force the time start from 0 and end with infinity now!")
        time_boundaries[0]=0
        time_boundaries[-1]=float('inf')
    return time_boundaries,IM_N1s,IM_N2s,m_t_s

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

def CDF_calculator(n, tmax, time_boundaries, m): #m(t) is piecewise constant given (left_)time_boundaries; tmax is the max t for plotting; n is the time interval
    CDF = []
    integ = 0
    for i in list(np.arange(0,tmax,n)):
        if i < time_boundaries[-1]:
            index = bisect.bisect_left(time_boundaries, i)
            m_i = m[index]
        else:
            m_i = m[-1]
        integ += 2 * m_i * n
        cdf = 1 - math.exp(-integ)
        CDF.append(cdf)
    return CDF

def getCDFintersect(t, CDF, val):
    xVec = t
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

time_boundaries,IM_N1s,IM_N2s,m_t_s = read_from_MSMC_IM(args.Input)
if args.samegrid:
    t = time_boundaries
    CDF_t = cumulative_Symmigproportion(time_boundaries, m_t_s)
    ofp=os.path.dirname(args.o)+'/{}.samegrid.plot_M_t.pdf'.format(os.path.basename(args.o))
else:
    t = list(np.arange(0,args.tmax,args.n))
    CDF_t = CDF_calculator(args.n, args.tmax, time_boundaries, m_t_s)
    ofp=os.path.dirname(args.o)+'/{}.tmax{}_n{}.plot_M_t.pdf'.format(os.path.basename(args.o),args.tmax, args.n)

xVec = [getCDFintersect(t, CDF_t, 0.25), getCDFintersect(t, CDF_t, 0.5), getCDFintersect(t, CDF_t, 0.75)]
yVec = [0.25, 0.5, 0.75]
plt.semilogx(t, CDF_t, label='M(t)', c='orange')
plt.stem(xVec, yVec, linefmt=':', basefmt=" ")
plt.xlabel("Generations ago", fontsize=12)
plt.ylabel("M(t)", fontsize=12)
plt.ylim((0,1))
plt.savefig(ofp,pad_inches = 0)
