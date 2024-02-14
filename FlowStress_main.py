# -*- coding: utf-8 -*-
"""
Robert Morden
University of Melbourne
February 2024

This code has been published as part of the following journal paper:
    Morden R, Horne A, Nathan R, Bond N R (2024), 
    XXXXXXXXXXXXXXXX

Calculating flow statistics and flow stress for MONTHLY data. Statistics are based on
Kennard et al (2010).

Kennard, M.J., Pusey, B.J., Olden, J.D., Mackay, S.J., Stein, J.L., Marsh, N., 2010.
Classification of natural flow regimes in Australia to support environmental flow management.
Freshwater Biology 55, 171–193. https://doi.org/10.1111/j.1365-2427.2009.02307.x

- This routine takes monthly flow data and calculates a range of statistics. It requires a BASELINE
  flow sequence as well as at least 1 SCENARIO sequence for comparison.
- As well as stats based on the full length of flow record, stats are also generated for 5 year
  blocks across the entire record. The distribution of 5 year block statistics is compared between
  the baseline and the scenario. The flow stress score is a measure of how different the two
  distributions are.
- A stress score of 0 means no statistically significant change between baseline and scenario.
- A stress score of 1 means the statistic has INCREASED entirely beyond its natural range.
- A stress score of -1 means the statistic has DECREASED entirely beyond its natural range.

- This file (FlowStress_main.py) is the core algorithm, opening input files and setting up calculations.
- The stats file (FlowStress_KennardIdicators.py) calculates the statistics and the 5 year blocks.
- The functions file (FlowStress_functions.py) contains a range of handy functions for some more
  repetitive aspects of the calculations, such as spells above/below a threshold, Colwell
  predictability calculations, or comparing distributions.

IMPORTANT NOTE
--------------

Not all indicators can be calculated using monthly data! Some simply do not translate
to a monthly timestep.
The final statistic results will not include results for:
    ML2, ML3
    DL1, Dl2, DL3, DL6, DL7, DL8
    DH1, DH2, DH3, DH6, DH7, DH8
    TL3, TL4
    TH3, TH4
    All RA statistics
    
Not all indicators can be calculated in 5 year blocks! Calculating the 1 year to 20 year ARI peak flows
is not possible with only 5 years of data.
The final stress score results will not include stress results for:
    MH9, MH10, MH11, MH12, MH13, MH14
    
"""

import pandas as pd
import numpy as np
import time
import warnings

import FlowStress_KennardIndicators as calc
import FlowStress_functions as calc2

warnings.simplefilter("error")

# Set up basic information ----------------------------------------------------------------------
"""
This section opens up the input files and gets everything ready to start
looping through sites.

This section can be rewritten as needed for different situations. For a big model run,
the file naming may be simple enough to allow the files to be opened programatically
instead of opening a list of scenarios.

"""
print('  Initialising') 
t = []                                                                         # initialise timings
t.append(time.time())

# get list of catchments
fname = 'FlowStress_catlist.csv'
cats = pd.read_csv(fname,index_col='catID')                                    # read catchment list
catareas = cats['area_km2']                                                    # extract catchment areas
excludelist = cats[cats['include'] == 'n'].index.to_list()                     # get list of catchments to EXCLUDE (if necessary)

# get scenario list
fname = 'FlowStress_scenlist.csv'
scens = pd.read_csv(fname,index_col='scen')
keyscens_list = scens.index.to_list()

# open baseline file
datestart = '01-jan-1891'
dateend = '31-dec-2019'
fname = 'FlowStress_flowbaseline.csv'
bflow = calc2.openflowdf(fname,excludelist,datestart,dateend)

loopreport = 1    # this is how many sites to calculate before updating progress on screen
                  # for big model runs, loopreport=100 is about right
                  # for short runs, loopreport=1 or 5 is fine

# calculate baseline flow stats -------------------------------------------------------------------
"""
This section loops through each site in the BASELINE flow file and calculates stats.
The baseline is done first so the 5 year block distributions are available later
when the scenario stats are calculated.

"""
print('  Calculating baseline monthly flow stats') 

# calculate baseline first so that sample distributions are available
for isite,site in enumerate(bflow.columns):
    
    if isite % loopreport == 0:
        print ('      Site ' + str(isite+1) + ' of ' + str(len(bflow.columns)) + time.strftime(' (%I:%M%p)'))
    
    # calculate baseline statistics
    site_stats,site_samples = calc.flowstats_mon(bflow[site],bflow[site],catareas[site])
    
    # tidy up results for this site
    site_stats['site'] = site                                                  # add site number to results
    site_stats['scen'] = 'bline'                                               # add scenario to results
    site_stats['cdf_diff'] = 0.0                                               # must be zero, this is the baseline!
    site_stats['stress'] = 0.0                                                 # must be zero, this is the baseline!
    site_samples.columns = ['{}_{}'.format(site,n) for n in site_samples.columns]
    
    # add results to global results dataframe
    if isite == 0:                                                             # first site - set up global results
        base_stats = site_stats
        base_samples = site_samples
    else:                                                                      # subsequent sites - append to global results
        base_stats = pd.concat([base_stats,site_stats])
        base_samples = pd.concat([base_samples,site_samples],axis=1)

t.append(time.time())
print('  Time to complete baseline stats (seconds) = ' + str(t[1] - t[0]))     # finalise

# calculate scenario flow stats -------------------------------------------------------------------
"""
This section calculates stats for scenarios. It loops through scenarios, then sites.
It also compares the 5 year block distributions to the baseline to get stress scores
for each site.

"""
print('  Calculating scenario monthly flow stats and stress scores') 

# calculate statistics and flow stress for all scenarios    
for iscen,scen in enumerate(keyscens_list):                                    # for each scenario
    
    print('    Scenario {}'.format(scen) + time.strftime(' (%I:%M%p)'))

    fname = scens.loc[scen,'flowfile']              # open monthly flow for scenario
    scnflow = calc2.openflowdf(fname,excludelist,datestart,dateend)

    for isite,site in enumerate(bflow.columns):                                # for each site
        
        if isite % loopreport == 0:
            print ('      Site ' + str(isite+1) + ' of ' + str(len(bflow.columns)) + time.strftime(' (%I:%M%p)'))

        # get stats (passed variables = scenario flow, baseline flow, catchment area)
        site_stats,site_samples = calc.flowstats_mon(scnflow[site],bflow[site],catareas[site])
        
        # tidy up results for this site
        site_stats['site'] = site                                              # add site number to results
        site_stats['scen'] = scen                                              # add scenario to results
        site_stats['cdf_diff'] = np.nan          # this is a temp entry, the correct number is obtained in the next section
        site_stats['stress'] = np.nan            # this is a temp entry, the correct number is obtained in the next section

        # calculate stress stats for this site
        for istat,stat in enumerate(np.unique(site_stats['stat'])):            # for each metric
            
            coltext = '{}_{}'.format(site,stat)
            
            aa = base_samples.loc[:,coltext].to_numpy()                        # get the base sample (column format = '22101_MA1')
            bb = site_samples.loc[:,stat].to_numpy()                           # get the site sample (column format = 'MA1')
            
            cond = (site_stats['stat'] == stat)                                # condition to get the right row in site_stats
            cond2 = (base_stats['stat'] == stat) & (base_stats['site'] == site)  # condition to get the right row in base_stats
            
            if stat in ['TL1','TH1']:                                          # special fix for metrics with circular means
                aa_shift = np.rint((6-base_stats.loc[cond2,'metric']).iloc[0]) # get "6 - baseline mean month" as an integer
                aa = np.where(aa+aa_shift>12,aa+aa_shift-12,
                              np.where(aa+aa_shift<1,aa+aa_shift+12,aa+aa_shift))  # need to adjust aa to put mean around 6
                bb = np.where(bb+aa_shift>12,bb+aa_shift-12,
                              np.where(bb+aa_shift<1,bb+aa_shift+12,bb+aa_shift))  # adjust bb by the same amount
            
            site_stats.loc[cond,'cdf_diff'] = calc2.ecdfoverlap(aa,bb)         # get the raw cdf overlap score
            site_stats.loc[cond,'stress'] = calc2.ecdfoverlap_ci(aa,bb,0.9)    # get the stress score
        
        if isite == 0:                                                         # add site results to dataframe (samples are discarded)
            scen_stats = site_stats
        else:
            scen_stats = pd.concat([scen_stats,site_stats])
        
    if iscen == 0:                                                             # add scenario results to dataframe
        all_stats = pd.concat([base_stats,scen_stats])
    else:
        all_stats = pd.concat([all_stats,scen_stats])
    
    
# calculate metric difference stats -----------------------------------------------------------------
"""
This section calculates the simple arithmetic difference between the scenario
statistic and the corresponding baseline.

"""
  
base_stats = all_stats.loc[all_stats['scen']=='bline',:].copy()
all_stats = all_stats.merge(base_stats,how='left',on=['stat','site'],suffixes=[None,'_baseline']).copy()
all_stats = all_stats.drop(columns=['scen_baseline','cdf_diff_baseline','stress_baseline'])
all_stats['diff'] = all_stats['metric']-all_stats['metric_baseline']

# save results and finish ---------------------------------------------------------------------------
"""
This section finishes the routine by saving files and finalising timestamps.
"""

t.append(time.time())
print('  Time to complete scenario stats (seconds) = ' + str(t[2] - t[1]))      # finalise

print('  Saving files')
all_stats.to_parquet('all_stats.par')  # save as a parquet file NOT csv!
                                       # Its much smaller and quicker to read later
# site_samples.to_csv('site_samples.csv')
# base_samples.to_csv('base_samples.csv')

t.append(time.time())  
print('  Finished!')
print('  Full execution time (s) = ' + str(t[3] - t[0]))
print('  Full execution time (m) = ' + str((t[3] - t[0])/60.0))
print('  Full execution time (h) = ' + str((t[3] - t[0])/3600.0))
