# MSMC-IM

This is a new method based on MSMC for inferring demographic parameters such as population size, split time, symmetric migration rates and when migration stops, by fitting MSMC results to a Isolation-Migration model. Currently there are two modes of the program: static and dynamic, which estimate either constant population size or time-dependent population size. 

# Getting Started
The program is written in python3. python 3 is required for running this program. 

# Guidance
Usage on MSMC_IM_StaticFit.py
```
usage: MSMC_IM_StaticFit.py [-h] [-N1 N1] [-N2 N2] [-NA NA] [-T T] [-t0 T0]
                            [-m M] [--Ti_MSMC] [-n_T N_T] [-N0 N0] [--noMig]
                            Input

Find parameters with the max likelihood for fitting IM model to MSMC

positional arguments:
  Input       OUTPUT from MSMC

optional arguments:
  -h, --help  show this help message and exit
  -N1 N1      Effective population size of Pop1. Default=1500
  -N2 N2      Effective population size of Pop2. Default=1500
  -NA NA      Effective population sieze of a population ancestral to
              population 1 and population 2. Default=1500
  -T T        Split time between population 1 and population 2. Default=2000
  -t0 T0      Time when migrations between population 1 and population 2 stop.
              Default=500
  -m M        Symetric Migration rate. Default=10e-5
  --Ti_MSMC   whether use the same time boundaries from MSMC for
              fitting(recommended). Default=False
  -n_T N_T    Number of time segments in total in fitting. Default=1000
  -N0 N0      Average effective population size. Default=20000
  --noMig     Option for estimating the migration rates or not(recommended).
              Default=False
```

Usage on MSMC_IM_DynamicFit.py
```
usage: MSMC_IM_DynamicFit.py [-h] [-N1 N1] [-N2 N2] [-NA NA] [-T T] [-t0 T0]
                             [-m M] [--Ti_MSMC] [-n_T N_T] [-N0 N0]
                             Input

Find parameters with the max likelihood for fitting IM model to MSMC

positional arguments:
  Input       OUTPUT from MSMC

optional arguments:
  -h, --help  show this help message and exit
  -N1 N1      Effective population size of Pop1. Default=1500
  -N2 N2      Effective population size of Pop2. Default=1500
  -NA NA      Effective population sieze of a population ancestral to
              population 1 and population 2. Default=1500
  -T T        Split time between population 1 and population 2. Default=2000
  -t0 T0      Time when migrations between population 1 and population 2 stop.
              Default=500
  -m M        Symetric Migration rate. Default=10e-5
  --Ti_MSMC   whether use the same time boundaries from MSMC for
              fitting(recommended).
  -n_T N_T    Number of time segments in total in fitting. Default=1000
  -N0 N0      Average effective population size. Default=20000

```
Here you can specify the initial values for parameters in the fitting. The required input file for MSMC_IM_StaticFit.py/MSMC_IM_DynamicFit.py is the MSMC result including six columns, as shown in the following:

```
time_index	left_time_boundary	right_time_boundary	lambda_00	lambda_01	lambda_11
0	-0	1.41401e-06	11901.4	0.0243435	0.0200401
1	1.41401e-06	2.86475e-06	22178.5	13.8383	9430.91
2	2.86475e-06	4.35418e-06	3119.07	145.393	11139.3
3	4.35418e-06	5.88443e-06	1830.22	689.113	2871.82
4	5.88443e-06	7.45778e-06	2222.18	2014.26	1806.34
5	7.45778e-06	9.07675e-06	3295.83	3356.35	3416.86
6	9.07675e-06	1.0744e-05	4809.94	5153.04	5496.14
7	1.0744e-05	1.24627e-05	6598.26	7268.54	7938.82
8	1.24627e-05	1.42358e-05	8488.37	9592.56	10696.7
...
```
