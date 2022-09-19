import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
import seaborn as sns
#from plotting import plotting_fcts
#from lib import get_nearest_neighbour
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio

# This script ...

# prepare data
data_path = "D:/Data/"

# check if folder exists
results_path = "../results/fluxnet/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
df_tmp = pd.read_csv(data_path + "FLUXNET/FLUXNET_SITE_ID_REDUCED-YY.csv", sep=',')
df_tmp = df_tmp.replace([-9999.0],np.nan)

dem_path = data_path + "DEMs/hyd_glo_dem_30s/hyd_glo_dem_30s.tif" #data_path + "DEMs/MERIT_250m/" + "dtm_elevation_merit.dem_m_250m_s0..0cm_2017_v1.0.tif"
dem = rio.open(dem_path, masked=True)

# average over years
df = df_tmp.groupby("SITE_ID").mean()
df = df.reset_index()

#df["LATENT HEAT FLUX"] = df["LATENT HEAT FLUX"]*12.87 # transform latent heat flux into ET using latent heat of vaporisation
#df["NET RADIATION"] = df["NET RADIATION"]*12.87

dem_data = dem.read().astype(float).flatten()
dem_data[dem_data < -1000] = np.nan
dem_data[dem_data > 10000] = np.nan
dem_data = dem_data[~np.isnan(dem_data)]

coord_list = [(x, y) for x, y in zip(df['LONGITUDE'], df['LATITUDE'])]
df['dem'] = [x for x in dem.sample(coord_list)]
df['dem'] = np.concatenate(df['dem'].to_numpy())
dem_observed = df["dem"]

# plot cdfs
# aridity
fig = plt.figure(figsize=(3, 2))
ax = plt.axes()
plt.grid(color='grey', linestyle='--', linewidth=0.25)
count, bins_count = np.histogram(dem_data, bins=10000)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], cdf, color="grey", label="Global distribution")
count, bins_count = np.histogram(dem_observed, bins=10000)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], cdf, color="black", label="Observations")
ax.set_xlim([0, 5000])
ax.set_ylim([0, 1])
# ax.set_xscale('log')
ax.set_xlabel("Elevation [m]")
ax.set_ylabel("Cumulative probability [-]")
plt.savefig(results_path + "CDF_elevation" + ".png", dpi=600, bbox_inches='tight')
plt.close()
print("Elevation")
print("Global distribution")
print("Fraction below 1000m: " + str(round(len(dem_data[dem_data<=1000]) / len(dem_data),2)))
print("Fraction above 1000m: " + str(round(len(dem_data[dem_data>1000]) / len(dem_data),2)))
print("Observations")
print("Fraction below 1000m: " + str(round(len(dem_observed[dem_observed<=1000]) / len(dem_observed),2)))
print("Fraction above 1000m: " + str(round(len(dem_observed[dem_observed>1000]) / len(dem_observed),2)))
