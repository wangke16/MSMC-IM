# MSMC-IM

MSMC-IM fits a continuous Isolation-Migration(IM) model with time-dependent population sizes N1(t) and N2(t), and a time-dependent continuous symmetric migration rate m(t) to the estimated coalescence rates from MSMC/MSMC2, which essentially is a re-parameterization from the triple of functions {λ11(t), λ12(t), λ22(t)} to a new triple of functions {N1(t), N2(t), m(t)}. To use MSMC-IM, you need to run either MSMC or MSMC2 to get an estimate on within-pop coalescence rates and across-pop coalescence rates. With MSMC output/combined MSMC2 output as input, we simply run MSMC-IM model by 
“MSMC_IM.py pair.combined.msmc2.txt > newestimates.output”. Also the time pattern needs to be specified, which is by default 1\*2+25\*1+1\*2+1\*3 as the default in MSMC2. In the output, MSMC-IM will rescale the scaled time in MSMC2 output by mutation rate 1.25e-8 into real time in generations, and report symmetric migration rates and M(t) in each time segment. 

# Getting Started
The program is written in python. Python 3 is required here. To use MSMC_IM.py, you have to import all neccessary functions written in MSMC_IM_funs.py. Make sure you download both python scripts for running MSMC-IM. 

# Guidance of usage on MSMC_IM.py
```
usage: MSMC_IM [-h] [-o O] [-N1 N1] [-N2 N2] [-m M] [-p P] [-beta BETA] 
               [--printfittingdetails] [--plotfittingdetails] [--xlog] 
               [--ylog]
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
  -beta BETA            Regularisation on estimating migration rat and population sizes. 
                        The bigger, the stronger penalty is.
                        Recommend: 1e-8,1e-6
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
You can find the example of input and output in the example filefolder. To get the output files in MSMC-IM/example/MSMC_IM_output, you need to run MSMC_IM.py using the input file Yoruba_French.8haps.combined.msmc2.final.txt by the following command line: 
```
MSMC_IM.py -beta 1e-8,1e-6 -o /MSMC-IM/example/MSMC_IM_output/Yoruba_French.8haps --printfittingdetails --plotfittingdetails --xlog /MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt
```
Penatly option ```--beta``` is always recommended while running modern human population pairs. 

You need to specify the directory and prefix of your output through ```option -o```, to generate the output ```.estimates.txt``` in the ceratin directory with your preferred prefix. The time boundaries is reported in generations in our ouput (gen=29 years). Here is an example of ```.estimates.txt```file: 
```
left_boundaries	im_N1	im_N2	m	M
0.0	198574.94595123047	39273.17391589546	3.9776767728370866e-25	0.0
69.80536	198574.94595123047	39273.17391589546	3.9776767728370866e-25	0.0
158.8328	198574.90056850316	39280.953360444604	3.9776767728370866e-25	0.0
272.3768	57003.32828207908	62637.81824920013	3.9776767728370866e-25	0.0
417.1872	26649.753206298024	30539.007130749862	3.9776767728370866e-25	0.0
...
```
Here we report M(t) evaluated at each time boundaries. As the cumulative probabiliy function of m(t), M(t) is a continuous distribtuion which can be evaluated at any given time t. ```plot_utils.py```enables you to plot M(t) at given specific time grid, simply by ```plot_utils.py -o /outdir/Yoruba_French.oranyprefix -tmax 100000 -n 100 /dir/to/Yoruba_French.estimates.txt``` to get ```/outdir/Yoruba_French.oranyprefix.plot_M_t.pdf```. 

You have the option for printing these internally estimated parameters while fitting through ```option --printfittingdetails```. If you would like to have a nice plot showing you how the fit looks like, you can plot these paramaters through ```option --plotfittingdetails```, and  ```option --xlog``` is always recommend. See ```.fittingdetails.txt``` and ```.fittingdetails.b21e-06.xlog.pdf``` in the /example/MSMC_IM_output/ filefolder. 
Here is an example of ```.fittingdetails.txt```file:
```
left_boundaries	IM_lambda00	IM_lambda01	IM_lambda11	IM_rCCR	MSMC_lambda00	MSMC_lambda01	MSMC_lambda11	MSMC_rCCR	msmc_N1	msmc_N2	naive_im_N1	naive_im_N2	im_N1	im_N2
0.0	2.51794101016e-06	0.0	1.27313366898e-05	0.0	2.3872624999999995e-06	3.0992375e-08	1.2872249999999997e-05	0.004062039989809636	209444.91860446855	38843.248072403825	198574.94595123047	39273.17391589546	198574.94595123047	39273.17391589546
69.80536	2.51794104905e-06	3.53345281497e-28	1.27313417205e-05	4.63425443461e-23	2.3872624999999995e-06	3.0992375e-08	1.2872249999999997e-05	0.004062039989809636	209444.91860446855	38843.248072403825	198574.94595123047	39273.17391589546	198574.94595123047	39273.17391589546
158.8328	2.51794168655e-06	8.03375725192e-28	1.27288313578e-05	1.05383050283e-22	2.3872624999999995e-06	2.4664625e-08	1.2872249999999997e-05	0.003232688462360774	209444.91860446855	38843.248072403825	198574.90056850316	39280.953360444604	198574.90056850316	39280.953360444604
272.3768	8.76519248136e-06	8.63705230112e-28	7.98671961388e-06	1.03117211361e-22	9.0226575e-06	2.465625e-08	7.7467375e-06	0.0029406248704857865	55416.045660604985	64543.299679381154	57003.32828207908	62637.81824920013	57003.32828207908	62637.81824920013
...
```
