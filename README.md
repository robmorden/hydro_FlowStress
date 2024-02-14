# hydro_FlowStress
Calculating flow statistics and flow stress based on monthly streamflow data.

This repository includes code for calculating flow statistics as used as part of the paper Morden et al (2024). The flow indicators themselves have been developed based on Kennard et al (2011). For further information please refer to the following two papers:

> Morden R, Horne A, Nathan R, Bond N R (2024), XXXXXXXXXXXXXXXX

> Kennard, M.J., Pusey, B.J., Olden, J.D., Mackay, S.J., Stein, J.L., Marsh, N., 2010. Classification of natural flow regimes in Australia to support environmental flow management. Freshwater Biology 55, 171–193. https://doi.org/10.1111/j.1365-2427.2009.02307.x

## Installation
I am a complete GitHub newbie as of August 2022. There is no prepared package to install, this is simply a repository of my code. If it looks useful, please download it and reference this repository or my paper Morden et al (2024).

## Overview
Flow statistics (or 'indicators') are commonly used to study how river ecosystems respond to flow regimes. In the simplest possible terms, ecological data such as species populations and diversity are sampled at regular time intervals, and this monitoring data is compared to flow data to observe any links between the two. This is called **flow ecology research**.

This code caclulates a suite of indicators using monthly flow data. The indicators were originally intended for daily data, but they have been translated where possible to use monthly data instead.

As well as calculating the indicators, the code also calculates flow stress. To do this, it compares results from a scenario flow sequence to results obtained from a baseline. Flow stress scores are calculated based on differences between the scenario and baseline. By calcuating indicators based on 5 year blocks of flow data, a number of 'sample' results can be obtained for each indicator, showing how it varies across a long period of record. Comparing the histograms of these samples between the scenario and baseline is what gives the stress score.

In short, a stress score of 0 means the scenario flow has not changed compared to the baseline. A score of 1 means the statistic has increased and is now 100% outside its natural range. A score of -1 means the statistic has decreased and is now 100% outside its natural range. Please refer to Morden et al (2024) for more information.

This code has been prepared to run sample data for demonstration purposes. The sample data includes flow data for 5 hypothetical catchments.

## Code structure
There are three code files:

* `FlowStress_main.py`  This file is the main wrapper function, opening all of the data files and calling functions to do all of the calculations.
* `FlowStress_KennardIndicators.py`  This file takes flow data for a single site and scenario and calculates flow statistics based on Kennard et al (2011) and Morden et al (2024).
* `FlowStress_functions.py`  This file contains a range of smaller functions which assist with the calculations. Many functions use the NUMBA library to speed them up.

## Inputs
The model requires basic information about catchments, scenarios, plus the baseline and scenario flow sequences. Each input is discussed below.
    
### List of catchments   
See sample data in the file `FlowStress_catlist.csv`.

The list of catchments must include the following fields:
                    
* `catID`      a unique alphanumeric ID for each catchment (should match the field name for the respective flow data)
* `area_km2`   area in km2
* `include`    should be set to either `y` or `n` to flag whether the catchment is to be calculated or skipped (useful for larger studies)
    
### List of scenarios
See sample data in the file `FlowStress_scenlist.csv`.

The list of catchments must include the following fields:
                    
* `scen`      a unique alphanumeric ID for each scenario (used to identify results in the final output file)
* `flowfile`  name of the file containing the flow data for that scneario. Flow data for multiple sites should be grouped into a single file per scenario.


### Baseline flow
Flow data in ML per month. See sample data in the file `FlowStress_flowbaseline.csv`.

### Scenario flow
Flow data in ML per month. See sample data in the file `FlowStress_flowscenarioA.csv`.

The format for both baseline and scenario flows should be as follows for the purposes of this code:

* Comments at the top of the file will be ignored if prefixed by the character `#`.
* `field 1`    date (month, year) in some sort of python interpretable format such as `1895-12-01`.
* `field 2`    site 1 flow (field name should match the unique id in catlist, see above)
* `field 3`    site 2 flow
etc...
