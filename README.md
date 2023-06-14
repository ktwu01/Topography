# The influence of topography on the global terrestrial water cycle

*Note: the paper has not been reviewed yet.*

This repository contains code used to make most of the plots shown in the review paper. Some plots were made using QGIS or taken from existing papers.

- plot_climatic_water_balance_elevation_profiles: plots P and PET against elevation for different rectangular swaths in different mountain ranges (Figure 2a)

- plot_swath_precipitation: plots P transect along swath (Figure 2b)

- plot_precipitation_uncertainty: plots P and PET transects for different forcing products in different regions (Figure 9 and Figure S2)

- analyse_Caravan: plots BFI and other baseflow signatures from Caravan against topographic slope (Figure 6a and Figure S6) and calculates fraction of observations per landform

- analyse_Fan: plots WTD against topographic slope (Figure S7) and calculates fraction of observations per landform

- analyse_Moeck: plots recharge against topographic slope and calculates fraction of observations per landform

- plot_DataBias: plots fration of landforms globally and fraction of observations per landform for Caravan, Fan, and Moeck data

- calculate_averages_per_landform: calculates average P and PE per landform.

- analyse_global_distributions: calculates global distributions of landforms and their slopes.

- resample_rasters: resamples and aligns different rasters

- functions: contains different helper functions for plotting and processing data

- data: contains data derived or used for some of the analyses

- QGIS: contains some infos on QGIS processing (QGIS files are stored locally)

- old: contains old scripts (not used for paper)

Links to the data sources can be found in the paper. 
