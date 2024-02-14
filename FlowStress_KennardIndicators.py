# -*- coding: utf-8 -*-
"""
Calculating flow statistics for MONTHLY data

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

IMPORTANT NOTE
--------------

Not all indicators can be calculated using monthly data!
The final statistic results will not include results for:
    ML2, ML3
    DL1, Dl2, DL3, DL6, DL7, DL8
    DH1, DH2, DH3, DH6, DH7, DH8
    TL3, TL4
    TH3, TH4
    All RA statistics
    
Not all indicators can be calculated in 5 year blocks!
The final stress score results will not include stress results for:
    MH9, MH10, MH11, MH12, MH13, MH14
    
"""

import pandas as pd
import numpy as np
from datetime import date
import scipy.stats as scistat

import FlowStress_functions as calc
from FlowStress_functions import scalardiv, seriesdiv #, arraydiv, meanORmed, trimseries

# ============================================================================
"""
Calculating a set of monthly flow statistics for a single timeseries

Input variables
---------------
qs    the scenario flow, the focus of the flow statistics
qb    the baseline flow, the reference series for comparison purposes
area  the catchment area

Variables passed to calculation routine
---------------------------------------
par   a set of parameters for the calculations (eg. how much flow is 'zero')
qm    monthly flow (scenario) with extra fields to assist with calculations
qy    yearly flow (scenario) with extra fields to assist with calculations
qbm   monthly flow (baseline) with extra fields to assist with calculations

All calculation results are assembled into a big dataframe of long term results
(results) and 5 year block results (samples).

I am sure there must be a better method to reassemble these results, but I am
just a self taught programming hack. This was what I came up with. It works.
"""

def flowstats_mon(qs,qb,area):
    
    if (qs.max() - qs.min() > 1):                                          # if the Q-scenario is a valid series (not all 0's or 5's)
        
        # set up some basic parameters ------------------------------------
        par = {}
        par['catarea'] = area                                              # catchment area in km2
        
        par['zeroflowday'] = 0.1                                           # 0.1 ML/d
        par['zeroflowmonth'] = 10                                          # 10 ML/m
        
        par['opt_nonzero'] = 0.1                                           # threshold for filtering out zero flow months/days
        
        par['opt_risefall'] = 'pcdiff'    #'absdiff'    #'pcdiff'          # option for rate of change calcs (absolute flow diff vs pc change)
        
        # set up some basic parameters and tidy dataframes ----------------
        qm = pd.DataFrame(qs)                                              # basic daily flow dataframe - scenario flow
        qm.columns = ['q']
        qm['qnz'] = qm['q'].where(cond=(qm['q']>=par['zeroflowmonth']),other=np.nan)  # non zero values
        qm['y'] = qm.index.year
        qm['m'] = qm.index.month
        
        qbm = pd.DataFrame(qb)                                             # basic daily flow dataframe - baseline flow
        qbm.columns = ['q']
        qbm['qnz'] = qbm['q'].where(cond=(qbm['q']>=par['zeroflowmonth']),other=np.nan)  # non zero values
        qbm['y'] = qbm.index.year
        qbm['m'] = qbm.index.month

        qy = pd.DataFrame(qs.resample('AS').sum())                         # aggregate scenario flow to annual for some stats
        qy.columns=['sum']
        qy['avg'] = qs.resample('AS').mean()
        qy['med'] = qs.resample('AS').median()
        qy['min'] = qs.resample('AS').min()
        qy['max'] = qs.resample('AS').max()
        qy['std'] = qs.resample('AS').std(ddof=1)
        qy['mmax'] = qs.resample('AS').max()
        qy['y'] = qy.index.year

        # Get magnitude stats (avg, low, high) ----------------------------
        ma, ml, mh, ma_s, ml_s, mh_s = calcmon_ma_ml_mh(qm,qy,qbm,par)

        # Low flow and high flow stats ------------------------------------
        fl, fh, fl_s, fh_s = calcmon_fl_fh(qm,qy,qbm,par)
        
        # Duration stats --------------------------------------------------
        dl, dh, dl_s, dh_s = calcmon_dl_dh(qm,qy,qbm,par)
        
        # # Timing stats ----------------------------------------------------
        ta, tl, th, ta_s, tl_s, th_s = calcmon_ta_tl_th(qm,qy,qbm,par)
        
        # Rise and fall stats ---------------------------------------------
        ra, ra_s = calcmon_ra(qm,qy,qbm,par)

        # combine stats into tidy dataframes ----------------------------
        ma_names = 'MA' + ma.index.astype(str)
        ml_names = 'ML' + ml.index.astype(str)
        mh_names = 'MH' + mh.index.astype(str)
        fl_names = 'FL' + fl.index.astype(str)
        fh_names = 'FH' + fh.index.astype(str)
        dl_names = 'DL' + dl.index.astype(str)
        dh_names = 'DH' + dh.index.astype(str)
        ta_names = 'TA' + ta.index.astype(str)
        tl_names = 'TL' + tl.index.astype(str)
        th_names = 'TH' + th.index.astype(str)
        ra_names = 'RA' + ra.index.astype(str)
        
        ma = pd.DataFrame({'stat':ma_names,'metric':ma})
        ml = pd.DataFrame({'stat':ml_names,'metric':ml})
        mh = pd.DataFrame({'stat':mh_names,'metric':mh})
        fl = pd.DataFrame({'stat':fl_names,'metric':fl})
        fh = pd.DataFrame({'stat':fh_names,'metric':fh})
        dl = pd.DataFrame({'stat':dl_names,'metric':dl})
        dh = pd.DataFrame({'stat':dh_names,'metric':dh})
        ta = pd.DataFrame({'stat':ta_names,'metric':ta})
        tl = pd.DataFrame({'stat':tl_names,'metric':tl})
        th = pd.DataFrame({'stat':th_names,'metric':th})
        ra = pd.DataFrame({'stat':ra_names,'metric':ra})
        
        # concat dataframes into 1 --------------------------------------
        results = pd.concat([ma, ml, mh,
                             fl, fh,
                             dl, dh, 
                             ta, tl, th, 
                             ra])
        
        # set up column names to be unique when concatenated -------------
        ma_s.columns = ['MA{}'.format(n) for n in ma_s.columns]
        ml_s.columns = ['ML{}'.format(n) for n in ml_s.columns]
        mh_s.columns = ['MH{}'.format(n) for n in mh_s.columns]
        fl_s.columns = ['FL{}'.format(n) for n in fl_s.columns]
        fh_s.columns = ['FH{}'.format(n) for n in fh_s.columns]
        dl_s.columns = ['DL{}'.format(n) for n in dl_s.columns]
        dh_s.columns = ['DH{}'.format(n) for n in dh_s.columns]
        ta_s.columns = ['TA{}'.format(n) for n in ta_s.columns]
        tl_s.columns = ['TL{}'.format(n) for n in tl_s.columns]
        th_s.columns = ['TH{}'.format(n) for n in th_s.columns]
        ra_s.columns = ['RA{}'.format(n) for n in ra_s.columns]
        
        # concat dataframes into 1 --------------------------------------
        samples = pd.concat([ma_s, ml_s, mh_s,
                             fl_s, fh_s,
                             dl_s, dh_s, 
                             ta_s, tl_s, th_s, 
                             ra_s],
                            axis=1)

    return results, samples

# ==================================================================================================================
# Calculating indices for average / low / high magnitudes
def calcmon_ma_ml_mh(qm,qy,qbm,par):
    
    # metrics
    ma = pd.Series(index=range(1,33),dtype='float64')
    ml = pd.Series(index=range(1,8),dtype='float64')
    mh = pd.Series(index=range(1,16),dtype='float64')

    # metric samples for comparing variability
    ma_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,33),dtype='float64')
    ml_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,8),dtype='float64')
    mh_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,16),dtype='float64')
    
    ind = 0
    
    mmbf = qbm['q'].mean()
    
    # annual stats (MA) ---------------------------------------------------------------------
    
    ma[1] = qm['q'].mean() / par['catarea']                                    # MMF divided by catchment area
    roll = qm['q'].rolling(window=60).mean().iloc[11::12]                      # 5 year moving mean, every 12th month from Dec1891
    ma_s.loc[:,1] = roll / par['catarea']
    
    ma[2] = qm['q'].median() / par['catarea']                                  # Median monthly flow divided by catchment area
    roll = qm['q'].rolling(window=60).median().iloc[11::12]
    ma_s.loc[:,2] = roll / par['catarea']
    
    ma[3] = qm['q'].std(ddof=1) / qm['q'].mean()                               # CV in monthly flows
    roll = qm['q'].rolling(window=60).std(ddof=1).iloc[11::12]
    roll2 = qm['q'].rolling(window=60).mean().iloc[11::12]
    ma_s.loc[:,3] = roll / roll2
        
    ma[4] = qm['q'].skew()                                                     # Skewness in monthly flows (g1 – Sokal & Rohlf, 1995)
    roll = qm['q'].rolling(window=60).skew().iloc[11::12]
    ma_s.loc[:,4] = roll
    
    qm_groups = qm.groupby('m')
    qm_monstats = qm_groups['q'].agg([np.mean,
                                      lambda x: np.std(x,ddof=1)])
    qm_monstats.columns = ['mean','sd']
    qm_monstats['cv'] = qm_monstats['sd']/qm_monstats['mean']
    
    ma.loc[5:16] = qm_monstats['mean'].values / mmbf                           # mean flow in each month / mmbf
    for i in range(5,17):
        ann = qm_groups['q'].get_group(i-4)
        ann.index = ma_s.index
        ma_s.loc[:,i] = ann / mmbf
        
    ma.loc[17:28] = qm_monstats['cv'].values                                   # cv of flow in each month
    for i in range(17,29):
        roll = qm_groups['q'].get_group(i-16).rolling(window=5).std(ddof=1)
        roll = roll / qm_groups['q'].get_group(i-16).rolling(window=5).mean()
        roll.index = ma_s.index
        ma_s.loc[:,i] = roll
   
        
    ma[29] = qy['sum'].mean() / par['catarea']                                 # Mean annual flow divided by catchment area
    ann = qy['sum'] / par['catarea']
    ann.index = ma_s.index
    ma_s.loc[:,29] = ann

    ma[30] = qy['sum'].std(ddof=1)/qy['sum'].mean()                            # CV in annual flows
    roll = qy['sum'].rolling(window=5).std(ddof=1) / qy['sum'].rolling(window=5).mean()
    roll.index = ma_s.index
    ma_s.loc[:,30] = roll
    
    ma[31] = qy['sum'].skew()                                                  # Skewness in annual flows (g1 – Sokal & Rohlf, 1995)
    roll = qy['sum'].rolling(window=5).skew()
    roll.index = ma_s.index
    ma_s.loc[:,31] = roll
    
    ma[32] = qy['sum'].median() / par['catarea']                               # Median annual flow divided by catchment area
    ann =  qy['sum'] / par['catarea']
    ann.index = ma_s.index
    ma_s.loc[:,32] = ann
    
    # low flow stats (ML) ---------------------------------------------------------------------
    
    ann = seriesdiv(qy['min'] , qy['avg'])
    ml[1] = ann.median()                                                       # Median of the lowest annual monthly flow divided by the MAMF averaged across all years 
                                                                               # My interpretation: median(MIN monthly each year / MEAN monthly each year)
    ann.index = ml_s.index
    ml_s.loc[:,1] = ann
    
    # bfd = calc.basefilter(qd['q'].to_numpy(),0.925)                            # daily baseflow series (alpha=0.925 is standard)
    # bfd = pd.Series(bfd,index=qd.index,name='q')   
    # bfy = bfd.resample('AS').sum()                                             # annual total baseflow
    # bfi = bfy/qy['sum']                                                        # annual ratio baseflow / total flow
    
    # ml[2] = bfi.mean()                                                         # Ratio of base flow to total flow, averaged across all years, where baseflow is calculated using 3-way digital filter (Grayson et al., 1996)
    # ml[3] = bfi.std(ddof=1) / bfi.mean()                                       # CV in Baseflow index
    
    ml[4] = qm['qnz'].quantile(0.25) / mmbf                                    # 75th percentile exceedance from the flow duration curve
    ml[5] = qm['qnz'].quantile(0.10) / mmbf                                    # 90th percentile exceedance from the flow duration curve
    ml[6] = qm['qnz'].quantile(0.01) / mmbf                                    # 99th percentile exceedance from the flow duration curve
    
    roll = qm['qnz'].rolling(window=60,min_periods=6).quantile(0.25).iloc[11::12]
    ml_s.loc[:,4] = roll / mmbf
    roll = qm['qnz'].rolling(window=60,min_periods=6).quantile(0.10).iloc[11::12]
    ml_s.loc[:,5] = roll / mmbf
    roll = qm['qnz'].rolling(window=60,min_periods=6).quantile(0.01).iloc[11::12]
    ml_s.loc[:,6] = roll / mmbf
    
    ml[7] = qy['min'].mean() / par['catarea']                                  # Mean annual minimum flow divided by catchment area
    ann = qy['min'] / par['catarea']
    ann.index = ml_s.index
    ml_s.loc[:,7] = ann
    
    # high flow stats (MH) ---------------------------------------------------------------------
    
    ann = qy['max'] / qy['avg']
    mh[1] = ann.median()                                                       # Median of the highest annual monthly flow divided by MAMF averaged across all years
                                                                               # My interpretation: median(max monthly each year / mean monthly flow each year)
    ann.index = mh_s.index
    mh_s.loc[:,1] = ann

    mh[2] = qm['qnz'].quantile(0.99) / mmbf                                    # 1st percentile exceedance from the flow duration curve
    mh[3] = qm['qnz'].quantile(0.90) / mmbf                                    # 10th percentile exceedance from the flow duration curve
    mh[4] = qm['qnz'].quantile(0.75) / mmbf                                    # 25th percentile exceedance from the flow duration curve
    
    roll = qm['qnz'].rolling(window=60,min_periods=6).quantile(0.99).iloc[11::12]
    mh_s.loc[:,2] = roll / mmbf
    roll = qm['qnz'].rolling(window=60,min_periods=6).quantile(0.90).iloc[11::12]
    mh_s.loc[:,3] = roll / mmbf
    roll = qm['qnz'].rolling(window=60,min_periods=6).quantile(0.75).iloc[11::12]
    mh_s.loc[:,4] = roll / mmbf

    mh[5] = qy['max'].mean() / par['catarea']                                  # Mean annual maximum flow divided by catchment area
    ann = qy['max'] / par['catarea']
    ann.index = mh_s.index
    mh_s.loc[:,5] = ann
    
    hfv1 = (qm['q'] - (1*mmbf)).clip(lower=0.0).resample('AS').sum()           # High Flow Volume (annual flows in excess of a threshold)
    hfv3 = (qm['q'] - (3*mmbf)).clip(lower=0.0).resample('AS').sum()           # High Flow Volume (annual flows in excess of a threshold)
    hfv7 = (qm['q'] - (7*mmbf)).clip(lower=0.0).resample('AS').sum()           # High Flow Volume (annual flows in excess of a threshold)
    
    mh[6] = hfv1.mean() / mmbf                                       # Mean of the high flow volume (calculated as the area between the hydrograph and the upper threshold defined as 1 times MMF) divided by MMF
    mh[7] = hfv3.mean() / mmbf                                       # Mean of the high flow volume (calculated as the area between the hydrograph and the upper threshold defined as 3 times MMF) divided by MMF
    mh[8] = hfv7.mean() / mmbf                                       # Mean of the high flow volume (calculated as the area between the hydrograph and the upper threshold defined as 7 times MMF) divided by MMF
    
    roll = hfv1.rolling(window=5).mean()
    roll.index = mh_s.index
    mh_s.loc[:,6] = roll / mmbf
    roll = hfv3.rolling(window=5).mean()
    roll.index = mh_s.index
    mh_s.loc[:,7] = roll / mmbf
    roll = hfv7.rolling(window=5).mean()
    roll.index = mh_s.index
    mh_s.loc[:,8] = roll / mmbf
    
    mh[9]  = calc.aepflow(qm['q'],1,'high','annual',ind) / mmbf                # Magnitude of 1 year ARI flood events (annual series).  Flood independence criteria = 7 days between peaks.
    mh[10] = calc.aepflow(qm['q'],2,'high','annual',ind) / mmbf                # Magnitude of 2 year ARI flood events (annual series).  Flood independence criteria = 7 days between peaks.
    mh[11] = calc.aepflow(qm['q'],5,'high','annual',ind) / mmbf                # Magnitude of 5 year ARI flood events (annual series).  Flood independence criteria = 7 days between peaks.
    mh[12] = calc.aepflow(qm['q'],10,'high','annual',ind) / mmbf               # Magnitude of 10 year ARI flood events (annual series).  Flood independence criteria = 7 days between peaks.
    mh[13] = calc.aepflow(qm['q'],15,'high','annual',ind) / mmbf               # Magnitude of 15 year ARI flood events (annual series).  Flood independence criteria = 7 days between peaks.
    mh[14] = calc.aepflow(qm['q'],20,'high','annual',ind) / mmbf               # Magnitude of 20 year ARI flood events (annual series).  Flood independence criteria = 7 days between peaks.

    mh[15] = qy['max'].skew()                                                  # Skewness in ann max flows (g1 – Sokal & Rohlf, 1995)
    roll = qy['max'].rolling(window=5).skew()
    roll.index = mh_s.index
    mh_s.loc[:,15] = roll

    return ma, ml, mh, ma_s, ml_s, mh_s

# ==================================================================================================================
# Calculating indices for frequencies of low / high events
def calcmon_fl_fh(qm,qy,qbm,par):

    fl = pd.Series(index=range(1,7),dtype='float64')
    fh = pd.Series(index=range(1,9),dtype='float64')
    
    fl_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,7),dtype='float64')
    fh_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,9),dtype='float64')

    qmq = qm['q'].to_numpy()
    qmy = qm['y'].to_numpy()
    qmm = qm['m'].to_numpy()

    ind = 0
    
    # Low flows (LF) ----------------------------------------------------------------------

    thresh = qbm['qnz'].quantile(0.25)                                         # 75th exceedance   <====
    starts,spellon,annstarts,anndays = calc.spellsbelow(qmq,qmm,qmm,qmy,thresh,ind) # spell stats
    fl[1] = fl[4] = 0.0
    fl_s.loc[:,1] = fl_s.loc[:,4] = 0
    if np.sum(annstarts)>0:
        fl[1] = np.mean(annstarts)                                             # if any spells, get average number per year
        fl_s.loc[:,1] = annstarts
    if np.sum(annstarts)>1:
        fl[4] = np.std(annstarts,ddof=1) / np.mean(annstarts)                  # if any spells, std x 100 / mean
        roll = fl_s.loc[:,1].rolling(window=5).std(ddof=1) / fl_s.loc[:,1].rolling(window=5).mean()
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        fl_s.loc[:,4] = roll

    thresh = qbm['qnz'].quantile(0.1)                                          # 90th exceedance   <====
    starts,spellon,annstarts,anndays = calc.spellsbelow(qmq,qmm,qmm,qmy,thresh,ind) # spell stats
    fl[2] = fl[5] = 0.0
    fl_s.loc[:,2] = fl_s.loc[:,5] = 0
    if np.sum(annstarts)>0:
        fl[2] = np.mean(annstarts)                                             # if any spells, get average number per year
        fl_s.loc[:,2] = annstarts
    if np.sum(annstarts)>1:
        fl[5] = np.std(annstarts,ddof=1) / np.mean(annstarts)                  # if any spells, std x 100 / mean
        roll = fl_s.loc[:,2].rolling(window=5).std(ddof=1) / fl_s.loc[:,2].rolling(window=5).mean()
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        fl_s.loc[:,5] = roll

    thresh = qbm['qnz'].quantile(0.01)                                         # 99th exceedance   <====
    starts,spellon,annstarts,anndays = calc.spellsbelow(qmq,qmm,qmm,qmy,thresh,ind) # spell stats
    fl[3] = fl[6] = 0.0
    fl_s.loc[:,3] = fl_s.loc[:,6] = 0
    if np.sum(annstarts)>0:
        fl[3] = np.mean(annstarts)                                             # if any spells, get average number per year
        fl_s.loc[:,3] = annstarts
    if np.sum(annstarts)>1:
        fl[6] = np.std(annstarts,ddof=1) / np.mean(annstarts)                  # if any spells, std x 100 / mean
        roll = fl_s.loc[:,3].rolling(window=5).std(ddof=1) / fl_s.loc[:,3].rolling(window=5).mean()
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        fl_s.loc[:,6] = roll

    # High flows (HF) ----------------------------------------------------------------------

    thresh = qbm['qnz'].quantile(0.75)                                         # 25th exceedance   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qmq,qmm,qmm,qmy,thresh,ind) # spell stats
    fh[1] = fh[4] = 0.0
    fh_s.loc[:,1] = fh_s.loc[:,4] = 0
    if np.sum(annstarts)>0:
        fh[1] = np.mean(annstarts)                                             # if any spells, get average number per year
        fh_s.loc[:,1] = annstarts
    if np.sum(annstarts)>1:
        fh[4] = np.std(annstarts,ddof=1) / np.mean(annstarts)                  # if any spells, std x 100 / mean
        roll = fh_s.loc[:,1].rolling(window=5).std(ddof=1) / fh_s.loc[:,1].rolling(window=5).mean()
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        fh_s.loc[:,4] = roll

    thresh = qbm['qnz'].quantile(0.9)                                          # 10th exceedance   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qmq,qmm,qmm,qmy,thresh,ind) # spell stats
    fh[2] = fh[5] = 0.0
    fh_s.loc[:,2] = fh_s.loc[:,5] = 0
    if np.sum(annstarts)>0:
        fh[2] = np.mean(annstarts)                                             # if any spells, get average number per year
        fh_s.loc[:,2] = annstarts
    if np.sum(annstarts)>1:
        fh[5] = np.std(annstarts,ddof=1) / np.mean(annstarts)                  # if any spells, std x 100 / mean
        roll = fh_s.loc[:,2].rolling(window=5).std(ddof=1) / fh_s.loc[:,2].rolling(window=5).mean()
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        fh_s.loc[:,5] = roll

    thresh = qbm['qnz'].quantile(0.99)                                          # 1st exceedance   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qmq,qmm,qmm,qmy,thresh,ind) # spell stats
    fh[3] = fh[6] = 0.0
    fh_s.loc[:,3] = fh_s.loc[:,6] = 0
    if np.sum(annstarts)>0:
        fh[3] = np.mean(annstarts)                                             # if any spells, get average number per year
        fh_s.loc[:,3] = annstarts
    if np.sum(annstarts)>1:
        fh[6] = np.std(annstarts,ddof=1) / np.mean(annstarts)                  # if any spells, std x 100 / mean
        roll = fh_s.loc[:,3].rolling(window=5).std(ddof=1) / fh_s.loc[:,3].rolling(window=5).mean()
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        fh_s.loc[:,6] = roll

    thresh = qbm['q'].mean() * 3                                                # mean monthly flow x 3   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qmq,qmm,qmm,qmy,thresh,ind) # spell stats
    fh[7] = 0.0
    fh_s.loc[:,7] = 0
    if np.sum(annstarts)>0:
        fh[7] = np.mean(annstarts)                                             # if any spells, get average number per year
        fh_s.loc[:,7] = annstarts
        
    thresh = qbm['q'].mean() * 7                                                # mean monthly flow x 7   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qmq,qmm,qmm,qmy,thresh,ind) # spell stats
    fh[8] = 0.0
    fh_s.loc[:,8] = 0
    if np.sum(annstarts)>0:
        fh[8] = np.mean(annstarts)                                             # if any spells, get average number per year
        fh_s.loc[:,8] = annstarts
        
    return fl, fh, fl_s, fh_s

# ==================================================================================================================
# Calculating indices for durations of low / high events
def calcmon_dl_dh(qm,qy,qbm,par):

    dl = pd.Series(index=range(1,19),dtype='float64')
    dh = pd.Series(index=range(1,17),dtype='float64')
    
    dl_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,19),dtype='float64')
    dh_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,17),dtype='float64')

    qmq = qm['q'].to_numpy()
    qmy = qm['y'].to_numpy()
    qmm = qm['m'].to_numpy()

    ind = 0
    
    mmbf = qbm['q'].mean()
    
    # rolling stats -----------------------------------------------------------

    min_1m = qm['q'].resample('AS').min()
    min_3m = qm['q'].rolling(3,center=False).mean().resample('AS').min()
    max_1m = qm['q'].resample('AS').max()
    max_3m = qm['q'].rolling(3,center=False).mean().resample('AS').max()
    
    min_1m.index = dl_s.index                                                  # adjust index to match dl_s and dh_s
    min_3m.index = dl_s.index
    max_1m.index = dl_s.index
    max_3m.index = dl_s.index

    min_1m = min_1m[1:]                                                        # [1:] strips out the first year where rolling mean fails
    min_3m = min_3m[1:]
    max_1m = max_1m[1:]
    max_3m = max_3m[1:]

    # low flow stats (DL) ----------------------------------------------------------
    # dl[1] = min_1d.mean()                                                      # Magnitude of minimum annual flows of various duration
    # dl[2] = min_3d[1:].mean()                                                  # [1:] strips out the first year where rolling mean fails
    # dl[3] = min_7d[1:].mean()
    dl[4] = min_1m.mean() / mmbf
    dl[5] = min_3m.mean() / mmbf
    
    dl_s.loc[:,4] = min_1m / mmbf
    dl_s.loc[:,5] = min_3m / mmbf
    
    # dl[6] = scalardiv(np.std(min_1d,ddof=1), np.mean(min_1d))                  # CV in magnitude of minimum annual flow of various duration
    # dl[7] = scalardiv(np.std(min_3d,ddof=1), np.mean(min_3d))
    # dl[8] = scalardiv(np.std(min_7d,ddof=1), np.mean(min_7d))
    dl[9] = scalardiv(np.std(min_1m,ddof=1), np.mean(min_1m))
    dl[10] = scalardiv(np.std(min_3m,ddof=1), np.mean(min_3m))
    
    dl_s.loc[:,9] = min_1m.rolling(window=5).std(ddof=1) / min_1m.rolling(window=5).mean()
    dl_s.loc[:,10] = min_3m.rolling(window=5).std(ddof=1) / min_3m.rolling(window=5).mean()
    
    thresh = qbm['qnz'].quantile(0.25)                                         # 75th exceedance   <====
    starts,spellon,annstarts,anndays = calc.spellsbelow(qmq,qmm,qmm,qmy,thresh,ind)    # spell stats
    dl[11] = dl[14] = 0.0
    dl_s.loc[:,11] = dl_s.loc[:,14] = 0.0
    if np.sum(annstarts)>0:
        df_starts = pd.DataFrame(starts,columns=['durations','intervals','year','month','day'],dtype='int32')
        df_starts.index = pd.to_datetime(df_starts.iloc[:,2:])
        df_ann_dur = df_starts['durations'].resample('AS',origin='1891-01-01').sum()
        df_ann_count = df_starts['durations'].resample('AS',origin='1891-01-01').count()
        roll = df_ann_dur.rolling(window=5).sum() / df_ann_count.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dl_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dl[11] = df_starts['durations'].mean()                                 # Mean duration of flows which remain below a lower threshold defined by the 75th exceedance percentile
        dl_s.loc[:,11] = roll
    if np.sum(annstarts)>1:
        dl[14] = df_starts['durations'].std(ddof=1) / df_starts['durations'].mean()  # CV in duration of annual occurrences
        roll = df_ann_dur.rolling(window=5).std(ddof=1) / df_ann_dur.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dl_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dl_s.loc[:,14] = roll
        
    thresh = qbm['qnz'].quantile(0.1)                                           # 90th exceedance   <====
    starts,spellon,annstarts,anndays = calc.spellsbelow(qmq,qmm,qmm,qmy,thresh,ind)    # spell stats
    dl[12] = dl[15] = 0.0
    dl_s.loc[:,12] = dl_s.loc[:,15] = 0.0
    if np.sum(annstarts)>0:
        df_starts = pd.DataFrame(starts,columns=['durations','intervals','year','month','day'],dtype='int32')
        df_starts.index = pd.to_datetime(df_starts.iloc[:,2:])
        df_ann_dur = df_starts['durations'].resample('AS',origin='1891-01-01').sum()
        df_ann_count = df_starts['durations'].resample('AS',origin='1891-01-01').count()
        roll = df_ann_dur.rolling(window=5).sum() / df_ann_count.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dl_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dl[12] = df_starts['durations'].mean()                                 # Mean duration of flows which remain below a lower threshold defined by the 75th exceedance percentile
        dl_s.loc[:,12] = roll
    if np.sum(annstarts)>1:
        dl[15] = df_starts['durations'].std(ddof=1) / df_starts['durations'].mean()  # CV in duration of annual occurrences
        roll = df_ann_dur.rolling(window=5).std(ddof=1) / df_ann_dur.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dl_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dl_s.loc[:,15] = roll

    thresh = qbm['qnz'].quantile(0.01)                                         # 99th exceedance   <====
    starts,spellon,annstarts,anndays = calc.spellsbelow(qmq,qmm,qmm,qmy,thresh,ind)    # spell stats
    dl[13] = dl[16] = 0.0
    dl_s.loc[:,13] = dl_s.loc[:,16] = 0.0
    if np.sum(annstarts)>0:
        df_starts = pd.DataFrame(starts,columns=['durations','intervals','year','month','day'],dtype='int32')
        df_starts.index = pd.to_datetime(df_starts.iloc[:,2:])
        df_ann_dur = df_starts['durations'].resample('AS',origin='1891-01-01').sum()
        df_ann_count = df_starts['durations'].resample('AS',origin='1891-01-01').count()
        roll = df_ann_dur.rolling(window=5).sum() / df_ann_count.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dl_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dl[13] = df_starts['durations'].mean()                                 # Mean duration of flows which remain below a lower threshold defined by the 75th exceedance percentile
        dl_s.loc[:,13] = roll
    if np.sum(annstarts)>1:
        dl[16] = df_starts['durations'].std(ddof=1) / df_starts['durations'].mean()  # CV in duration of annual occurrences
        roll = df_ann_dur.rolling(window=5).std(ddof=1) / df_ann_dur.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dl_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dl_s.loc[:,16] = roll

    countall = qm['q'].resample('AS').count()
    countzeros = countall - qm['qnz'].resample('AS').count()
    countzeros.index = dl_s.index
    
    dl[17] = countzeros.mean()                                                 # Mean annual number of months having zero flow
    dl_s.loc[:,17] = countzeros
    dl[18] = calc.scalardiv(countzeros.std(ddof=1), countzeros.mean())         # CV in annual number of months having zero flow 
    roll = countzeros.rolling(window=5).std(ddof=1) / countzeros.rolling(window=5).mean()
    roll = roll.iloc[4:].replace([np.inf,np.nan],0)
    dl_s.loc[:,18] = roll
    
    # high flow stats (DH) -------------------------------------------------------------------
    # dh[1] = max_1d.mean()                                                      # Magnitude of maximum annual flows of various duration
    # dh[2] = max_3d.mean()
    # dh[3] = max_7d.mean()
    dh[4] = max_1m.mean() / mmbf
    dh[5] = max_3m.mean() / mmbf

    dh_s.loc[:,4] = max_1m / mmbf
    dh_s.loc[:,5] = max_3m / mmbf

    # dh[6] = np.std(max_1d,ddof=1) / np.mean(max_1d)                            # CV in magnitude of maximum annual flows of various duration
    # dh[7] = np.std(max_3d,ddof=1) / np.mean(max_3d)
    # dh[8] = np.std(max_7d,ddof=1) / np.mean(max_7d)
    dh[9] = np.std(max_1m,ddof=1) / np.mean(max_1m)
    dh[10] = np.std(max_3m,ddof=1) / np.mean(max_3m)
        
    dh_s.loc[:,9] = max_1m.rolling(window=5).std(ddof=1) / max_1m.rolling(window=5).mean()
    dh_s.loc[:,10] = max_3m.rolling(window=5).std(ddof=1) / max_3m.rolling(window=5).mean()
    
    thresh = qbm['qnz'].quantile(0.75)                                         # 25th exceedance   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qmq,qmm,qmm,qmy,thresh,ind)   # spell stats
    dh[11] = dh[14] = 0.0
    dh_s.loc[:,11] = dh_s.loc[:,14] = 0.0
    if np.sum(annstarts)>0:
        df_starts = pd.DataFrame(starts,columns=['durations','intervals','year','month','day'],dtype='int32')
        df_starts.index = pd.to_datetime(df_starts.iloc[:,2:])
        df_ann_dur = df_starts['durations'].resample('AS',origin='1891-01-01').sum()
        df_ann_count = df_starts['durations'].resample('AS',origin='1891-01-01').count()
        roll = df_ann_dur.rolling(window=5).sum() / df_ann_count.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dh_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dh[11] = df_starts['durations'].mean()                                 # Mean duration of flows which remain below a lower threshold defined by the 75th exceedance percentile
        dh_s.loc[:,11] = roll
    if np.sum(annstarts)>1:
        dh[14] = df_starts['durations'].std(ddof=1) / df_starts['durations'].mean()  # CV in duration of annual occurrences
        roll = df_ann_dur.rolling(window=5).std(ddof=1) / df_ann_dur.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dh_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dh_s.loc[:,14] = roll
    
    thresh = qbm['qnz'].quantile(0.90)                                         # 10th exceedance   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qmq,qmm,qmm,qmy,thresh,ind)   # spell stats
    dh[12] = dh[15] = 0.0
    dh_s.loc[:,12] = dh_s.loc[:,15] = 0.0
    if np.sum(annstarts)>0:
        df_starts = pd.DataFrame(starts,columns=['durations','intervals','year','month','day'],dtype='int32')
        df_starts.index = pd.to_datetime(df_starts.iloc[:,2:])
        df_ann_dur = df_starts['durations'].resample('AS',origin='1891-01-01').sum()
        df_ann_count = df_starts['durations'].resample('AS',origin='1891-01-01').count()
        roll = df_ann_dur.rolling(window=5).sum() / df_ann_count.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dh_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dh[12] = df_starts['durations'].mean()                                 # Mean duration of flows which remain below a lower threshold defined by the 75th exceedance percentile
        dh_s.loc[:,12] = roll
    if np.sum(annstarts)>1:
        dh[15] = df_starts['durations'].std(ddof=1) / df_starts['durations'].mean()  # CV in duration of annual occurrences
        roll = df_ann_dur.rolling(window=5).std(ddof=1) / df_ann_dur.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dh_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dh_s.loc[:,15] = roll
    
    thresh = qbm['qnz'].quantile(0.99)                                         # 1st exceedance   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qmq,qmm,qmm,qmy,thresh,ind)   # spell stats
    dh[13] = dh[16] = 0.0
    dh_s.loc[:,13] = dh_s.loc[:,16] = 0.0
    if np.sum(annstarts)>0:
        df_starts = pd.DataFrame(starts,columns=['durations','intervals','year','month','day'],dtype='int32')
        df_starts.index = pd.to_datetime(df_starts.iloc[:,2:])
        df_ann_dur = df_starts['durations'].resample('AS',origin='1891-01-01').sum()
        df_ann_count = df_starts['durations'].resample('AS',origin='1891-01-01').count()
        roll = df_ann_dur.rolling(window=5).sum() / df_ann_count.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dh_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dh[13] = df_starts['durations'].mean()                                 # Mean duration of flows which remain below a lower threshold defined by the 75th exceedance percentile
        dh_s.loc[:,13] = roll
    if np.sum(annstarts)>1:
        dh[16] = df_starts['durations'].std(ddof=1) / df_starts['durations'].mean()  # CV in duration of annual occurrences
        roll = df_ann_dur.rolling(window=5).std(ddof=1) / df_ann_dur.rolling(window=5).sum()
        roll.index = pd.date_range(date(roll.index[0].year,12,1),date(roll.index[-1].year,12,1),freq='12MS')
        roll = roll.reindex(dh_s.index)
        roll = roll.iloc[4:].replace([np.inf,np.nan],0)
        dh_s.loc[:,16] = roll
 
    return dl, dh, dl_s, dh_s

# ==================================================================================================================
# Calculating indices for timing and predictability of flows
def calcmon_ta_tl_th(qm,qy,qbm,par):

    
    ta = pd.Series(index=range(1,5),dtype='float64')
    tl = pd.Series(index=range(1,5),dtype='float64')
    th = pd.Series(index=range(1,5),dtype='float64')
    
    ta_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,5),dtype='float64')
    tl_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,5),dtype='float64')
    th_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,5),dtype='float64')

    qmq = qm['q'].to_numpy()
    # qmmin = qm['min'].to_numpy()             # can't get min flow in each month
    # qmmax = qm['max'].to_numpy()             # can't get max flow in each month
    qmm = qm['m'].to_numpy()
    qmy = qm['y'].to_numpy()
    
    # setting up colwell calculations ----------------------------------------
    
                # ===========================
                # for Colwell calcs, create 11 bins as follows:
                #    bin 6 = 20 x mean daily flow
                #    all bins are double the size of the one before
                # ===========================
                
    meanmonflow = qbm['q'].mean()
    
    bins = np.array([0, 0.625, 1.25, 2.5, 5, 10, 20, 40, 80, 160, 320, 9999])
    bins = bins * meanmonflow                                                  # multiply values by mean monthly flow
    
    # annual timing (TA) -----------------------------------------------------
    
    p,c,m = calc.colwell(qmq,bins,qmm,12)                                      # get colwell stats
    
    ta[1] = p                                                                  # predictability
    ta[2] = c                                                                  # constancy
    ta[3] = m/p                                                                # contingencey (as proportion of predictability)

    rollwindow = 60                                                            # unfortunately, pandas rolling windows do not
    rollinterval = 12                                                          # allow the .apply method to use my Colwell                                                                 # routine, so I have to do it myself...
    p,c,m = calc.rolling_colwell(qmq,bins,qmm,12,rollwindow,rollinterval)
    
    ta_s.iloc[4:,0] = p
    ta_s.iloc[4:,1] = c
    ta_s.iloc[4:,2] = np.asarray(m)/np.asarray(p)
    
    # Revised code for TA4
    yr_ratio = calc.min6m_ratio(qmq,qmy)
    ta[4] = np.mean(yr_ratio)                                                    # PC contribution of 6 driest months to discharge in each calendar year
    ta_s.loc[:,4] = yr_ratio               
        
    # low flow timing (TL) ----------------------------------------------------
    
    annmin = qm.groupby('y')['q'].agg(['min','idxmin'])
    annmin['m'] = pd.Index(annmin['idxmin']).month
    idx = pd.to_datetime(pd.DataFrame({'year':annmin.index}).assign(month=12).assign(day=1))
    annmin.index = idx
    
    tl[1] = scistat.circmean(annmin['m'].to_numpy(),high=12,low=1)             # mean julian date of the min flow
    tl[2] = scistat.circstd(annmin['m'].to_numpy(),high=12,low=1)              # SD (not CV any more!) of julian date of the min flow
    
    roll = annmin['m'].rolling(window=5).apply(lambda x: scistat.circstd(x.to_numpy(),high=12,low=1))
    tl_s.loc[:,1] = annmin['m']
    tl_s.loc[:,2] = roll
    
    # p,c,m = calc.colwell(qmmin,bins,qmm,12)                                    # get colwell stats
    
    # tl[3] = p                                                                  # predictability
    # tl[4] = m/p                                                                # contingencey (as proportion of predictability)
    
    # high flow timing (TH) ---------------------------------------------------
    
    annmax = qm.groupby('y')['q'].agg(['max','idxmax'])
    annmax['m'] = pd.Index(annmax['idxmax']).month
    idx = pd.to_datetime(pd.DataFrame({'year':annmax.index}).assign(month=12).assign(day=1))
    annmax.index = idx
    
    th[1] = scistat.circmean(annmax['m'].to_numpy(),high=12,low=1)             # mean julian date of the max flow
    th[2] = scistat.circstd(annmax['m'].to_numpy(),high=12,low=1)              # SD (not CV any more!) of julian date of the max flow
    
    roll = annmax['m'].rolling(window=5).apply(lambda x: scistat.circstd(x.to_numpy(),high=12,low=1))
    th_s.loc[:,1] = annmax['m']
    th_s.loc[:,2] = roll

    # p,c,m = calc.colwell(qmmax,bins,qmm,12)                                    # get colwell stats
    
    # th[3] = p                                                                  # predictability
    # th[4] = m/p                                                                # contingencey (as proportion of predictability)
    
    return ta, tl, th, ta_s, tl_s, th_s

# ==================================================================================================================
# Calculating indices for rise and fall rates
def calcmon_ra(qm,qy,qbm,par):

    ra = pd.Series(index=range(1,7),dtype='float64')

    ra_s = pd.DataFrame(index=pd.date_range('1891/12/1','2019/12/1',freq='12MS'),columns=range(1,7),dtype='float64')

    # rise and fall ---------------------------------------------------------------------
    # qdq = qd['q'].to_numpy()
    # qdy = qd['y'].to_numpy()

    # chngrates,chngpc,numrevs,flipdays,annflips = calc.risefall(qdq,qdy)        # rise and fall stats
    
    # rates = chngpc                                                             # 'rates' by default means pc change (today - prevday) / prevday
    # if par['opt_risefall'] == 'absdiff':                                       # option to make 'rates' = absolute diff from prev day (today - prevday)
    #     rates = chngrates
    # ratesr = rates[rates>0]
    # ratesf = rates[rates<0]
    
    # ra[1] = np.mean(ratesr)                                                    # avg rise rate
    # ra[2] = np.std(ratesr,ddof=1) / np.mean(ratesr)                            # cv of rise rates
    
    # ra[3] = -1 * np.mean(ratesf)                                               # avg fall rate, modified to ensure result is always >=0
    # ra[4] = -1 * np.std(ratesf,ddof=1) / np.mean(ratesf)                       # cv of fall rates
    
    # ra[5] = meanORmed(annflips,par['opt_median'])                              # average number of flips/yr
    # ra[6] = np.std(annflips,ddof=1) / np.mean(annflips)                        # cv of flips/yr
    
    return ra, ra_s





