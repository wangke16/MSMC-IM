# MSMC-IM

This is a new method based on MSMC for inferring demographic parameters such as population size, split time, symmetric migration rates and when migration stops, by fitting MSMC results to a Isolation-Migration model. Currently there are two modes of the program: static and dynamic, which estimate either constant population size or time-dependent population size. 
# Guidance
Usage: MSMC_IM_StaticFit.py [-h] [-N1 N1] [-N2 N2] [-NA NA] [-T T] [-t0 T0]
                            [-m M] [--Ti_MSMC] [-n_T N_T] [-N0 N0] [--noMig]
                            Input

Usage: MSMC_IM_DynamicFit.py [-h] [-N1 N1] [-N2 N2] [-NA NA] [-T T] [-t0 T0]
                             [-m M] [--Ti_MSMC] [-n_T N_T] [-N0 N0]
                             Input
# Installation
The program is written in python3. python 3 is required for running this program. 
