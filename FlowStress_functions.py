# -*- coding: utf-8 -*-
"""
Flow statistics handy functions

Robert Morden
University of Melbourne
February 2024

This code has been published as part of the following journal paper:
    Morden R, Horne A, Nathan R, Bond N R (2024), 
    XXXXXXXXXXXXXXXX

Trying to use Numba wherever possible

These functions were written to assist with the calculation of flow indicators.
They are provided 'warts and all'.

If you find them useful, please give credit as required. Thanks!

"""

import numpy as np
from numba import jit
import pandas as pd
import scipy.stats as scistat
import scipy.special as scispec
import math

# ==================================================================================================================
# Annual series
# calculate annual vols, mins, maxs, minmonth/day, maxmonth/day
# Works for MONTHLY or DAILY
#@jit(nopython=True)
def annual_stats(flow,dy,dstep):                             # dstep is either the month number (1-12) or the day number (1 to 366)

    numflows = len(flow)
    allyears = np.unique(dy)                                 # get a list of unique years
    numyears = len(allyears)

    annvol = np.zeros(numyears)                              # 1 column, 1 row per year
    anncvs = np.zeros(numyears)                              # 1 column, 1 row per year
    annstd = np.zeros(numyears)                              # 1 column, 1 row per year
    annavg = np.zeros(numyears)                              # 1 column, 1 row per year
    
    annmax = np.zeros(numyears)                              # 1 column, 1 row per year
    annmin = np.full(numyears,9999999.0)                     # 1 column, 1 row per year, need to initialise with a big number
    annmaxstep = np.zeros(numyears)                          # 1 column, 1 row per year
    annminstep = np.zeros(numyears)                          # 1 column, 1 row per year
    
    for i in range(numflows):
        thisyear = dy[i]
        thisstep = dstep[i]
        idxthisyear = np.where(allyears==thisyear)[0][0]     # 'where' returns a tuple of numpy arrays - need array entry 1 of tuple entry 1
        
        if flow[i] < annmin[idxthisyear]:
            annmin[idxthisyear] = flow[i]
            annminstep[idxthisyear] = thisstep
        if flow[i] > annmax[idxthisyear]:
            annmax[idxthisyear] = flow[i]
            annmaxstep[idxthisyear] = thisstep
    
    for i in range(numyears):
        thisyear = allyears[i]
        cond = (dy==thisyear)
        annvol[i] = np.sum(flow[cond])
        annavg[i] = np.mean(flow[cond])
        annstd[i] = std_s(flow[cond],ddof=1)
        if annavg[i]>0:
            anncvs[i] = scalardiv(annstd[i] , annavg[i])
        else:
            anncvs[i] = 0.0
        
    return annvol,annmax,annmin,annmaxstep,annminstep,anncvs

# ==================================================================================================================
# Seasonal series
# calculate 12 monthly means, medians, mins, maxs
# Works for MONTHLY only
@jit(nopython=True)
def seasonal_stats(flow,dm):
    
    seasavg = np.zeros(12)
    seasmed = np.zeros(12)
    seasmax = np.zeros(12)
    seasmin = np.zeros(12)
    
    for i in range(12):
        seas = flow[dm==i+1]
        seasavg[i] = np.mean(seas)
        seasmed[i] = np.median(seas)
        seasmax[i] = np.max(seas)
        seasmin[i] = np.min(seas)
    
    return seasavg,seasmed,seasmax,seasmin

# ==================================================================================================================
# Annual series for each month
# calculate 12 monthly means, medians, mins, maxs in ML/d or ML/m
# Works for DAILY only 
@jit(nopython=True)
def monthly_stats(flow,dy,dm):
    
    allyears = np.unique(dy)                                   # get a list of unique years
    numyears = len(allyears)
    
    mavg = np.zeros((numyears,12))                             # set up results arrays
    mmed = np.zeros((numyears,12))
    mmin = np.zeros((numyears,12))
    mmax = np.zeros((numyears,12))
    mcv = np.zeros((numyears,12))
    
    for y in range(numyears):
        for m in range(12):
            thismonth = flow[(dy==allyears[y]) * (dm==m+1)]    # slice to get a single month of flows
            mavg[y,m] = np.mean(thismonth)                     # get stats for that month
            mmed[y,m] = np.median(thismonth)
            mmin[y,m] = np.min(thismonth)
            mmax[y,m] = np.max(thismonth)
            if mavg[y,m]==0:
                mcv[y,m] = 0.0
            else:
                mcv[y,m] = std_s(thismonth,ddof=1) / mavg[y,m] * 100
        
    return mavg,mmed,mmax,mmin,mcv                             # each variable is numyears x 12

# ==================================================================================================================
# Standard deviation for a sample (n-1)
# Using NUMBA, the usual "std" function cannot accept any arguments.
# This means that it defaults to the population std where ddof=0
# This routine gives the std where ddof can be specified.
# Inarray = input numpy array (1d)
# ddof = degrees of freedom, typically 1 for a hydrological sample
@jit(nopython=True)
def std_s(inarray,ddof):
    n = len(inarray)
    mean_s = np.mean(inarray)
    sum_sq_diff = np.sum((inarray - mean_s)**2)
    sd = np.sqrt(sum_sq_diff / (n-ddof))
    return sd

# ==================================================================================================================
# Spells below for a single 1d series
# Works for MONTHLY or DAILY
@jit(nopython=True)
def spellsbelow(flows,dd,dm,dy,threshold,ind):
    
    numflows = len(flows)                              # basic stats
    allyears = np.unique(dy)
    numyears = len(allyears)
    
    dur = []                                           # output variables
    intafter = []
    starty = []
    startm = []
    startd = []
    spellon = np.zeros(numflows)
    annstarts = np.zeros(numyears)
    anndays = np.zeros(numyears)
    
    ievent = -1                                          # looping variables
    prevflow = 999999999.0
    indTF = True
    
    for i in range(numflows):
        thisyear = dy[i]
        iyr = np.where(allyears==thisyear)[0][0]        # 'where' returns a tuple of numpy arrays - need array entry 1 of tuple entry 1
        if flows[i] <= threshold:                       # spell event in progress
            if prevflow > threshold:                        # spell event just started
                if indTF:                                       # if it IS independant
                    ievent = ievent + 1                             # +1 spell
                    dur.append(0)                                   # add a duration entry
                    intafter.append(0)                              # add an interval entry
                    starty.append(dy[i])                            # record start date
                    startm.append(dm[i])
                    startd.append(dd[i])
                    annstarts[iyr] = annstarts[iyr] + 1             # +1 start for this year

                else:                                           # if its NOT independent
                    dur[ievent] = dur[ievent] + intafter[ievent]    # add the recent interval back into the spell duration
                    anndays[iyr] = anndays[iyr] + intafter[ievent]  # add the recent interval into the spell days count for this year
                    intafter[ievent] = 0                            # reset the interval to zero
                    
            spellon[i] = 1                                  # mark this day as having active spell
            dur[ievent] = dur[ievent] + 1                   # +1 duration
            anndays[iyr] = anndays[iyr] + 1                 # +1 spell day for this year
        
        elif ievent >= 0:                               # if there has been at least 1 spell
            intafter[ievent] = intafter[ievent] + 1         # +1 to the interval after the event
            if intafter[ievent] >= ind:                     # spell is independent when interval is long enough
                indTF = True
            else:
                indTF = False
            
        prevflow = flows[i]
        
    starts = np.asarray([dur,intafter,starty,startm,startd]).T
    
    return starts, spellon, annstarts, anndays

# ==================================================================================================================
# Spells above for a single 1d series
# Works for MONTHLY or DAILY
@jit(nopython=True)
def spellsabove(flows,dd,dm,dy,threshold,ind):
    
    numflows = len(flows)                                 # basic stats
    allyears = np.unique(dy)
    numyears = len(allyears)
    
    dur = []                                              # output variables
    intafter = []
    qvol = []
    qmax = []
    starty = []
    startm = []
    startd = []
    spellon = np.zeros(numflows)
    annstarts = np.zeros(numyears)
    anndays = np.zeros(numyears)

    ievent = -1                                           # looping variables
    prevflow = 0.0
    indTF = True
    
    for i in range(numflows):
        thisyear = dy[i]
        iyr = np.where(allyears==thisyear)[0][0]          # 'where' returns a tuple of numpy arrays - need array entry 1 of tuple entry 1
        if flows[i] > threshold:                          # spell event in progress
            if prevflow <= threshold:                         # spell event just started
                if indTF:                                         # if it IS independant
                    ievent = ievent + 1                               # +1 spell
                    dur.append(0)                                     # add a duration entry
                    qvol.append(0.0)                                  # add an event volume entry
                    qmax.append(0.0)                                  # add an event max flow entry
                    intafter.append(0)                                # add an interval entry
                    starty.append(dy[i])                              # record start date
                    startm.append(dm[i])
                    startd.append(dd[i])
                    annstarts[iyr] = annstarts[iyr] + 1               # +1 start for this year
                
                else:                                             # if its NOT independent
                    dur[ievent] = dur[ievent] + intafter[ievent]      # add the recent interval back into the spell duration
                    anndays[iyr] = anndays[iyr] + intafter[ievent]    # add the recent interval into the spell days count for this year
                    intafter[ievent] = 0                              # reset the interval to zero
                    
            spellon[i] = 1                                # mark this day as having active spell
            dur[ievent] = dur[ievent] + 1.0                   # +1 duration
            qvol[ievent] = qvol[ievent] + flows[i]            # aggregate volume for this event
            qmax[ievent] = max(qmax[ievent],flows[i])         # get peak vol for this event
            anndays[iyr] = anndays[iyr] + 1                   # +1 spell day for this year
        
        elif ievent >= 0:                             # if there has been at least 1 spell
            intafter[ievent] = intafter[ievent] + 1       # +1 to the interval after the event
            if intafter[ievent] >= ind:                   # spell is independent when interval is long enough
                indTF = True
            else:
                indTF = False
                
        prevflow = flows[i]                           # reset
    
    starts = np.asarray([dur,intafter,starty,startm,startd]).T
    
    return starts, qvol, qmax, spellon, annstarts, anndays

# ==================================================================================================================
# Flow percentiles
# based on an array of percentiles and a flow series
# Works for MONTHLY or DAILY
@jit(nopython=True)
def flowpercentiles(flows,parray):
    
    if len(flows)==0:
        pflows = parray*0.0
    elif max(flows) < 0.1:
        pflows = parray*0.0
    else:
        parray_int = 100-parray
        pflows = np.percentile(flows,parray_int)
    
    return pflows

# ==================================================================================================================
# Calculate rates of rise and fall
# Assumed to be averaged across the entire record, dates not required
@jit(nopython=True)
def risefall(flows,dy):
    
    numflows = len(flows)                                                      # basic stats
    allyears = np.unique(dy)
    numyears = len(allyears)

    annflips = np.zeros(numyears)                                              # annflips is the number of reversals in each caendar year
    diff = np.zeros(numflows)                                                  # diff is the increase from the prev step in ML
    rate = np.zeros(numflows)                                                  # rate is the proportional change (increase from the prev timestep divided by the prev flow)
    change = np.zeros(numflows)                                                # change is a rise or fall flag (1 or -1 only)
    flip = np.zeros(numflows)                                                  # flip is set to 1 if there is a reversal
    
    for i in range(1,numflows):                                                # first step is ignored
        thisyear = dy[i]
        iyr = np.where(allyears==thisyear)[0][0]    # 'where' returns a tuple of np arrays - need array entry 1 of tuple entry 1

        diff[i] = flows[i] - flows[i-1]                                        # calculate diff
        if flows[i-1] == 0:
            rate[i] = 0.0
        else:
            rate[i] = diff[i] / flows[i-1]

        if i==1:                                                               # on the first loop
            diff[0] = diff[1]                                                  # initialise the diff to avoid a flip
            rate[0] = rate[1]                                                  # initialse rate to match the second timestep
            if rate[0]<0:                                                      # initialise change to match the second timestep
                change[0] = -1
            elif rate[0]>0:
                change[0] = 1
            else:
                change[0] = -1                                                 # if the second timestep is stady, then assume falling
                
        if rate[i] < 0:                                                        # if falling
            change[i] = -1
        elif rate[i] > 0:                                                      # if rising
            change[i] = 1
        else:                                                                  # if steady then continue from the prev timestep
            change[i] = change[i-1]
        
        if change[i] != change[i-1]:                                           # if change not equal to the prev timestep, its a reversal
            flip[i] = 1
            annflips[iyr] = annflips[iyr] + 1
            
    reversals = np.sum(flip)                         # count flips
    
    return diff,rate,reversals,flip,annflips

# ==================================================================================================================
# Predictability, Constancy, Contingency
# Based on    Colwell RK. 1974. Predictability, Constancy, and Contingency of Periodic Phenomena. Ecology 55: 1148–53.
# I know numpy can do 2D histograms, but we'll see if numba can speed it up
# Inputs - flows=array of flow values, bins=numpy style upper bound of flow categories, seasons=array of months (1-12) or days(1-365)
@jit(nopython=True)
def colwell(flows,bins,seasons,nseasons):
    
    s = len(bins)-1                                                            # number of states, less one to account for the end bin value
    t = nseasons                                                               # number of seasons
    
    hist2d = np.zeros((s,t))                                                   # set up histogram array (r x c = states x seasons)
    #h_array = np.zeros((s,t))                                                  # set up prob array
    flowbins = np.digitize(flows,bins,right=False)                             # assign bin number to each flow
                                                                                  ## 'right=False' means that zero flows
                                                                                  ## will be assigned to the first bin if the
                                                                                  ## first value in the bin variable is zero.
    
    for i in range(len(flows)):                                                # 2d histogram of flows (bins/seasons)
        if seasons[i]<=nseasons:                                                 # if day=366, just skip it
            irow = flowbins[i]-1
            icol = seasons[i]-1
            hist2d[irow,icol] = hist2d[irow,icol] + 1  
    
    z = hist2d.sum()
    
    hx = 0.0
    for mnth in range(t):                                                      # sum counts per season, apply log eqn, sum = HX
        ratio = hist2d[:,mnth].sum() / z
        if ratio > 0:
            hx = hx - (ratio * np.log(ratio))
    
    hy = 0.0
    for state in range(s):                                                     # sum counts per state, apply log eqn, sum = HY
        ratio = hist2d[state,:].sum() / z
        if ratio > 0:
            hy = hy - (ratio * np.log(ratio))
    
    hxy = 0.0
    for irow in range(s):                                                         # sum individual counts, apply log eqn, sum = HXY
        for icol in range(t):
            val = hist2d[irow,icol]
            if val > 0:
                ratio = val / z
                hxy = hxy - (ratio * np.log(ratio))
        
    m = (hx + hy - hxy) / np.log(s)
    c = 1 - (hy / np.log(s))
    p = m + c
    
    return p,c,m    # p = predictibility, c = constancy, m = contingency
    
# ==================================================================================================================
# Rolling window colwell calcs
@jit(nopython=True)
def rolling_colwell(flows,bins,seasons,nseasons,rollwindow,rollinterval):
    
    numflows = len(flows)
    
    results_p = []
    results_c = []
    results_m = []
    
    for istart in range(0,numflows-rollwindow+1,rollinterval):
        iend = istart+rollwindow
        p,c,m = colwell(flows[istart:iend], bins, seasons[istart:iend], 12)
        results_p.append(p)
        results_c.append(c)
        results_m.append(m)

    return results_p, results_c, results_m

# ==================================================================================================================
"""
Calculations for flood or low flow with a specific AEP
Method based on HIT manual
  q = daily flow (pandas SERIES with full datetime index - NOT a dataframe! )
  ari = flood annual recurrance interval
  highlow = 'high' or 'low' depending on whether high or low flows are being calculated (only works for annual series)
  method = 'annual' or 'partial'
  ind = independance, used to select peaks over threshold for partial series (only needed for partial series)
NOTE - for an annual series calculation, set the 1yr ARI flow to the min of the annual series
NOTE - Annual series calcs can work with monthly data 
DO NOT USE @JIT - JUST USE PANDAS INSTEAD, its easier to get the peaks and sort them out
"""
def aepflow(q,ari,highlow,method,ind):
    
    if method == 'annual':
        
        if highlow == 'high':
            z = scistat.norm.ppf(1-(1/ari))                                    # get the z value
            annseries = q.resample('AS').agg(['max','idxmax'])                 # get the annual series
            annseries['log'] = annseries['max'].clip(lower=0.1)
            annseries['log'] = np.log10(annseries['log'])                      # get the log 10 of peaks (avoid log zero)
        
        else:
            z = scistat.norm.ppf(1/ari)                                        # get the z value
            annseries = q.resample('AS').agg(['min','idxmin'])                 # get the annual series
            annseries['log'] = annseries['min'].clip(lower=0.1)
            annseries['log'] = np.log10(annseries['log'])                      # get the log 10 of peaks (avoid log zero)

        m = annseries['log'].mean()
        sd = annseries['log'].std(ddof=1)
        
        # Estimate the T=1.67 ARI flood (Qt) from this series from x = M + Z*SD
        # where the Z value corresponding to an AEP of 1/1.67 = norm.s.inv(1-1/1.67)
        #                                                     = -0.25
        # thus Qt = M-0.25*SD

        qt = 10**(m + (sd * z))
        
        if ari==1:
            if highlow == 'high':                                              # but if the ari=1, just get the min of annual series
                qt = annseries['max'].min()
            else:
                qt = annseries['min'].max()
    
    elif method == 'partial':    

        n = len(np.unique(q.index.year))                                       # number of years
 
        window = (ind*2)+1                                                     # define window
        rollmax = q.rolling(window=window,center=True).max()                   # rolling window (2xind), fix to centre, calculate peak
        peaks = pd.Series(True,index=q.index,name='peak')                      # create dataframe of 'True' values
        peaks = peaks.where(cond=(rollmax==q),other=False)                     # if the rolling max centre = q, then flag as peak
        
        pot = q[peaks]                                                         # select only flagged days
        m = min(len(pot),n)                                                    # number of peaks to select = min (avail peaks, n)
        mpot = pot.nlargest(m).sort_values(ascending=True)                     # select m peaks over a threshold (='mpot')
            
        # Partial series flood estimation using method of L moments (ARR book 3, Chapter 2) ----------------
        # These calcs are based on Tony Ladsons blog and spreadsheet:
        # https://tonyladson.wordpress.com/2019/03/25/fitting-a-probability-model-to-pot-data/
        
        mpot = pd.DataFrame(mpot)
        mpot['i'] = range(1,m+1)                                               # number
        mpot['rank'] = m - mpot['i'] + 1                                       # rank in descending order
        mpot['i-1'] = mpot['i'] - 1
        mpot['n-i'] = n - mpot['i']
        mpot['diff'] = mpot['i-1'] - mpot['n-i']
        mpot['diff_x_q'] = mpot['diff'] * mpot['q']
        
        lambda1 = mpot['q'].mean()                                             # L moment 1
        lambda2 = mpot['diff_x_q'].sum() / scispec.comb(m,2) / 2               # L moment 2
        beta = lambda2 * 2                                                     # prediction coefficient
        qstar = lambda1 - beta                                                 # prediction coefficient
        
        mpot['ey'] = (mpot['rank']-0.4) / (n+0.2)                               # ey = expected prob in 1 year = plotting position
        mpot['ari'] = 1 / mpot['ey']                                           # ari for each flow
        mpot['qfit'] = qstar - (beta * np.log(mpot['ey']))                    # fitted values of q based on the model
        
        # get qt -----------------------------------
        ey = 1/ari
        qt = qstar - (beta * math.log(ey))                                     # predict using coefficients and natural log of ey
        
    return qt

# ==================================================================================================================
# Digital filter for baseflow separation
# Works for DAILY only
# based on method 3 (pp 78) from Hydrological Recipes (Grayson et al 1996)
# 'flows' must be a numpy array
# Always applies 3 passes - forward, backward, forward
# NOTE - On each timestep, the total flow and quick flow are initialised by assuming
# the previous timestep is exactly the same (probably not far from reality).
@jit(nopython=True)
def basefilter(flows,alpha):
    n = len(flows)
    qf = np.zeros(len(flows))
    qb = np.zeros(len(flows)) 
    
    qtot = flows.copy()                                                               # pass 1
    qprev = qtot[0]
    qfprev = qtot[0] * (1-alpha)     # set initial baseflow = 7.5% of total flow
    for i in range(n):
        q = qtot[i]
        qf[i] = (alpha * qfprev)  +  ( (q-qprev) * (1+alpha) / 2 )
        if qf[i] < 0:
            qb[i] = q
        else:
            qb[i] = q-qf[i]
        qprev = q
        qfprev = qf[i]

    qtot = qb.copy()                                                                  # pass 2 (reverse)
    qprev = qtot[-1]
    qfprev = qtot[-1] * (1-alpha)     # set initial baseflow = 7.5% of total flow
    for i in range(n-1,-1,-1):        # I would use reversed(range(n)), but @jit does not support it
        q = qtot[i]
        qf[i] = (alpha * qfprev)  +  ( (q-qprev) * (1+alpha) / 2 )
        if qf[i] < 0:
            qb[i] = q
        else:
            qb[i] = q-qf[i]
        qprev = q
        qfprev = qf[i]
        
    qtot = qb.copy()                                                                  # pass 3
    qprev = qtot[0]
    qfprev = qtot[0] * (1-alpha)     # set initial baseflow = 7.5% of total flow
    for i in range(n):
        q = qtot[i]
        qf[i] = (alpha * qfprev)  +  ( (q-qprev) * (1+alpha) / 2 )
        if qf[i] < 0:
            qb[i] = q
        else:
            qb[i] = q-qf[i]
        qprev = q
        qfprev = qf[i]

    return qb        


# ==================================================================================================================
# Some index calculations have a user option for mean OR median
def meanORmed(arr,opt):                   
    
    if opt == 'median':
        return np.median(arr)
    elif opt =='mean':
        return np.mean(arr)

# ==================================================================================================================
# Trim a daily flow series to remove NaNs and partial years at the start and end
# Output should be a series which starts on 1st Jan and ends on 31st Dec with no NaNs
def trimseries(qin):
    
    qout = qin.dropna()                                                        # get the portion of the series without NaNs
    
    if qout.index[0].month > 1:                                                # trim to whole calendar years
        newstartdate = pd.to_datetime(str(qout.index[0].year + 1) + '-01-01')
        qout = qout[newstartdate:]
    if qout.index[-1].month < 12:
        newenddate = pd.to_datetime(str(qout.index[-1].year - 1) + '-12-31')
        qout = qout[:newenddate]
    
    return pd.Series(qout)                                                     # return a series trimmed to whole calendar years

# ==================================================================================================================
# Skewness g1
# Input is a single pandas series 'x'
def skew_g1(x):                   

    '''
    g1 formula is from:
    
        Jolicoeur P. (1999) The skewness and peakedness indices, g1 and g2. In: Introduction to Biometry.
        Springer, Boston, MA. https://doi.org/10.1007/978-1-4615-4777-8_14
    
    It suggests that g1 = k3 / k2**1.5
               where k2 = (1/(N-1)) * sum((x-xbar)**2)
                 and k3 = (N/((N-1)(N-2))) * sum((x-xbar)**3)  where all equations are for samples.
    '''
    
    xbar = x.mean()
    n = len(x)
    k2 = (1/(n-1)) * ((x-xbar)**2).sum()
    k3 = (n/((n-1)*(n-2))) * ((x-xbar)**3).sum()
    g1 = k3 / (k2**1.5)

    return g1

# ==================================================================================================================
# Calculate ratio of flow in min 6 months for each year
# Works for MONTHLY only
@jit(nopython=True)
def min6m_ratio(flow,dy):
    
    allyears = np.unique(dy)                              # basic stats
    numyears = len(allyears)
    
    result = np.zeros(numyears)
    
    for y in range(numyears):
        thisyear = flow[(dy==allyears[y])]                       # slice to get a single year of flows
        sumyear = np.sum(thisyear)
        sum6min = np.sum(np.partition(thisyear,6)[:6])
        result[y] = sum6min/sumyear
        
    return result

# ==================================================================================================================
"""
Empirical cumulative distribution function (Ecdf) difference calculator

aa, bb = numpy arrays of input values

Values in aa and bb are used to define their respective cdf's
cdf's are defined here like an exceedance curve, with p=0 for xmin and p=1 for xmax. Function is smooth, not stepped.

The values from bb are interpolated onto the cdf for aa to get old and new probabilities
  - in effect, this is plotting the two cdfs, and calculating the vertical differences for each value of bb

The absolute difference in probablities is summed and divided by n/2 to get a stress score between 0 and 1
  - for example, if all bb values were lower than the minimum of aa, then
    their p values on the aa cdf would be 0. So their differences would add up to n/2
  - so need to divide by n/2 to ensure the maximum possible score is 1.

The sign of the stress score is based on the median of aa and bb

"""
@jit(nopython=True)
def ecdfoverlap(aa,bb):
    
    # get sorted unique values from each array (aa_x) and assign probabilities (aa_p)
    # cdf = prop aa values which are <= individual aa_x values
    # cdf starts at 0, ends at 1
    # use cdf as a smooth function, like an exceedance curve
    aa_nonan = aa[~np.isnan(aa)]                                               # remove nans
    bb_nonan = bb[~np.isnan(bb)]
    
    result = np.nan
    if len(aa_nonan)*len(bb_nonan) > 0:
        aa_x = np.sort(aa_nonan)                                                   # sort small to large
        bb_x = np.sort(bb_nonan) 
        n_aax = len(aa_x)                                                          # get n
        n_bbx = len(bb_x)
        aa_cdf = np.linspace(0,1,n_aax)                                            # cdf for aa (regularly spaced increments from 0 to 1)
        bb_cdf = np.linspace(0,1,n_bbx)                                            # cdf for bb (regularly spaced increments from 0 to 1)
    
        # make aa and bb monotonic by adding a small increment to non-unique values
        aam_x = monotonic(aa_x,0.01)
        bbm_x = monotonic(bb_x,0.01)
        
        # interpolate bb values onto the aa cdf
        bb_cdf_on_aa = np.interp(bbm_x,aam_x,aa_cdf)                               # interpolate bb_x onto aa_x, return aa_cdf value
            
        # get the differences 
        bb_aa_cdf_diff = np.abs(bb_cdf_on_aa - bb_cdf)
        
        result = np.sum(bb_aa_cdf_diff) / (n_bbx / 2)                              # sum the proportions and divide by n/2
        
        if np.median(aa_x) > np.median(bb_x):                                      # sign is based on median difference
            result = abs(result) * -1.0
        else:
            result = abs(result)
    
    return result

# ==================================================================================================================
"""
Empirical cumulative distribution function (Ecdf) difference calculator (uses Confidence Intervals)

aa, bb = numpy arrays of input values

Values in aa and bb are used to define their respective cdf's
cdf's are defined here like an exceedance curve, with p=0 for xmin and p=1 for xmax. Function is smooth, not stepped.

The values from bb are interpolated onto the cdf for aa to get old and new probabilities
  - in effect, this is plotting the two cdfs, and calculating the vertical differences for each value of bb

The absolute difference in probablities is summed and divided by n/2 to get a stress score between 0 and 1
  - for example, if all bb values were less than the minimum of aa, then
    their p values interpolated on the aa cdf would be 0. So their differences would add up to n/2
  - so need to divide by n/2 to ensure the maximum possible score is 1.

ONLY BB VALUES OUTSIDE OF THE AA CONFIDENCE INTERVALS ARE INCLUDED

The sign of the stress score is based on the median of aa and bb

"""
@jit(nopython=True)
def ecdfoverlap_ci(aa,bb,ci_width):
    
    # get sorted unique values from each array (aa_x) and assign probabilities (aa_p)
    # cdf = prop aa values which are <= individual aa_x values
    # cdf starts at 0, ends at 1
    # use cdf as a smooth function, like an exceedance curve
    aa_nonan = aa[~np.isnan(aa)]                                               # remove nans
    bb_nonan = bb[~np.isnan(bb)]
    
    result = np.nan
    if min(len(aa_nonan),len(bb_nonan)) > 3:                                   # if both aa and bb have at least 3 entries
        aa_x = np.sort(aa_nonan)                                               # sort small to large
        bb_x = np.sort(bb_nonan) 
        n_aax = len(aa_x)                                                      # get n
        n_bbx = len(bb_x)
        aa_cdf = np.linspace(0,1,n_aax)                                        # cdf for aa (regularly spaced increments from 0 to 1)
        bb_cdf = np.linspace(0,1,n_bbx)                                        # cdf for bb (regularly spaced increments from 0 to 1)
    
        # make aa and bb monotonic by adding a small increment to each value (affine transformation)
        aam_x = monotonic(aa_x,1e-5)
        bbm_x = monotonic(bb_x,1e-5)
        
        # interpolate bb values onto the aa cdf
        bb_cdf_on_aa = np.interp(bbm_x,aam_x,aa_cdf)                           # interpolate bb_x onto aa_x, return aa_cdf value
        
        # get the 95% confidence intervals
        # code based on https://james-brennan.github.io/posts/edf/
        # confidence interval based on Dvoretzky–Kiefer–Wolfowitz (DKW) inequality
        # (https://en.wikipedia.org/wiki/Dvoretzky%E2%80%93Kiefer%E2%80%93Wolfowitz_inequality)
        alpha = 1 - ci_width                      # if 90% confidence, then alpha = 10%
        e = np.sqrt(1.0/(2*len(aam_x)) * np.log(2./alpha))
        
        # get the differences 
        bb_aa_cdf_diff = np.abs(bb_cdf_on_aa - bb_cdf)
        
        for i in range(len(bb_cdf)):
            if bb_aa_cdf_diff[i] <= e:                                         # if difference is within confidence intervals,
                bb_aa_cdf_diff[i] = 0.0                                        # then set to zero
        
        result = np.sum(bb_aa_cdf_diff) / (n_bbx / 2)                          # sum the proportions and divide by n/2
        
        if np.median(aa_x) > np.median(bb_x):                                  # sign is based on median difference
            result = abs(result) * -1.0
        else:
            result = abs(result)
    
    return result

# ==================================================================================================================
# Adjust a series of values to ensure they are monotonically increasing
# Adds a small amount to EVERY X VALUE to ensure that each value is larger
# than the previous one.
# The value alpha specifies the maximum amount to add.
# alpha = 0.00001 (1e-5) is suggested to ensure that the added increments do not materially
# change the series of values.
# ASSUME THAT aa HAS ALL NUMERIC VALUES, NO NANS OR INFS ALLOWED!!
@jit(nopython=True)
def monotonic(aa,alpha):
    
    aa = np.sort(aa)                                                           # sort aa from smallest to largest
    n_aa = len(aa)
    d = alpha / (n_aa-1)                                                       # increment = alpha / n-1  
        
    # set up results array
    aa_mono = aa.astype('float64')                                             # change to a float, just in case the input is al integers
    
    # find the next increase beyond the repeat values (d1)
    for icount,x in enumerate(aa):                                             # loop through every value in aa
        increment = icount * d
        aa_mono[icount] = x + increment                                        # aa + increment
    
    return aa_mono

# ==================================================================================================================
# Safely divide SCALARS without an error - if the divisor is zero return zero
def scalardiv(num,div):                   
    
    if div == 0:
        result = 0
    else:
        result = num / div
        
    return result

# ==================================================================================================================
# Safely divide ARRAYS without an error - if the divisor is zero return zero
def arraydiv(aa,bb):                   
    
    aa[aa==0] = np.nan
    bb[bb==0] = np.nan
    result = aa / bb
    result[np.isnan(result)] = 0.0
    return result

# ==================================================================================================================
# Safely divide Panda series without an error - if the divisor is zero return zero
def seriesdiv(aa,bb):                   
    
    result = aa.div(bb.replace(0,np.nan)).fillna(0.0)
    return result

# ==================================================================================================================
# Short routine to open a flow file and set up a tidy dataframe
def openflowdf(fname,sitestoexclude,dstart,dend):
    
    outflow = pd.read_csv(fname,                                               # read in MONTHLY flow data
                          na_values=-99.99,
                          comment='#',
                          index_col=0,
                          parse_dates=True)
    outflow = outflow.drop(columns=['year','month'])                           # drop the year and month columns

    # get included sites
    outflow = outflow.drop(columns=sitestoexclude)                             # filter sites to include
    
    # trim dates
    outflow = outflow.loc[dstart:dend,:]
    
    return outflow




