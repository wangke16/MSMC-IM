# MSMC-IM

MSMC-IM fits a continuous Isolation-Migration(IM) model with time-dependent population sizes N1(t) and N2(t), and a time-dependent continuous symmetric migration rate m(t) to the estimated coalescence rates from MSMC/MSMC2, which essentially is a re-parameterization from the triple of functions {λ11(t), λ12(t), λ22(t)} to a new triple of functions {N1(t), N2(t), m(t)}. To use MSMC-IM, you need to run either MSMC or MSMC2 to get an estimate on within-pop coalescence rates and across-pop coalescence rates. With MSMC output/combined MSMC2 output as input, we simply run MSMC-IM model by 
“MSMC_IM.py pair.combined.msmc2.txt > newestimates.output”. Also the time pattern needs to be specified, which is by default 1*2+25*1+1*2+1*3 as the default in MSMC2. In the output, MSMC-IM will rescale the scaled time in MSMC2 output by mutation rate 1.25e-8 into real time in generations, and report symmetric migration rates and M(t) in each time segment. 

# Getting Started
The program is written in python. Python 3 is required here. To use MSMC_IM.py, you have to import all neccessary functions written in MSMC_IM_funs.py. Make sure you download both python scripts for running MSMC-IM. 

# Guidance of usage on MSMC_IM.py
```
usage: MSMC_IM [-h] [-o O] [-N1 N1] [-N2 N2] [-m M] [-p P]
               [--beta BETA [BETA ...]] [--printfittingdetails]
               [--plotfittingdetails] [--xlog] [--ylog]
               Input

Estimate time-dependetn migration rates through fitting IM model to coalescent
rates from MSMC

positional arguments:
  Input                 Time dependent within-pop coalescent rates and cross-
                        pop coalescent rates for a pair of populations, e.g. t
                        wopops.combined.msmc2.final.txt/twopops.msmc.final.txt

optional arguments:
  -h, --help            show this help message and exit
  -o O                  output directory and prefix of output
  -N1 N1                Initial constant effective population size of Pop1 to
                        start fitting process. Default=15000
  -N2 N2                Initial constatnt effective population size of Pop2 to
                        start fitting process. Default=15000
  -m M                  Initial symmetric migration rate between two pops to
                        start fitting process. Default=0
  -p P                  Pattern of fixed time segments
                        [default=1*2+25*1+1*2+1*3(MSMC2)], which has to be
                        consistent with MSMC2 or MSMC(default=10*1+15*2)
                        output you are using here
  --beta BETA [BETA ...]
                        Regularisation on estimated migration rate and
                        population size. The bigger, the stronger penalty is.
                        Recommend: 0,1e-6
  --printfittingdetails
                        Print detailed infomation during fitting process e.g.
                        estimated split time from M(t) midpoint, initial and
                        final Chi-Square value, estimated coalesent rates from
                        IM model. Defaul=False
  --plotfittingdetails  Plot IM estiamtes on popsize, coalescent
                        rates, in contrast to MSMC estimates. Default=False
  --xlog                Plot all parameters in log scale on x-axis.
                        Default=False. Recommend to add this flag.
  --ylog                Plot all parameters in log scale on y-axis.
                        Default=False
```
You can find the example of input and output in the example filefolder. To get the output files in MSMC-IM/example/output, you need to run MSMC_IM.py using the input file Yoruba_French.8haps.combined.msmc2.final.txt by the following command line: 
```
MSMC_IM.py --beta 0,1e-6 -o .git/MSMC-IM/example/output/Yoruba_French.8haps --printfittingdetails --plotfittingdetails --xlog .git/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt
```
Penatly option ```--beta``` is always neccessary for fitting process based on our tests on world-wide popuations. 

You always need to specify the directory and prefix of your output through ```option -o```, to generate the output ```.estimates.txt``` in the ceratin directory with your preferred prefix. ```.estimates.txt```file looks like this: 
```
time_index	left_time_boundary	right_time_boundary	m	M
0	0.0	69.80536	4.430565528189666e-25	0.0
1	69.80536	158.8328	4.430565528189666e-25	0.0
2	158.8328	272.3768	4.430565528189666e-25	0.0
3	272.3768	417.1872	4.430565528189666e-25	0.0
4	417.1872	601.8752000000001	4.430565528189666e-25	0.0
...
```
You have the option for printing these internally estimated parameters used in fitting and reporting through ```option --printfittingdetails```. If you would like to have a nice plot showing you how the fit looks like, you can plot these paramaters through ```option --plotfittingdetails```, and  ```option --xlog``` is always recommend. See ```.fittingdetails.txt``` and ```.fittingdetails.b10.0.b21e-06.xlog.pdf``` in the example.
 
