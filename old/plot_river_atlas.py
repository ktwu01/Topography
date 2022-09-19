import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import pyosp
import fiona
import geopandas as gpd
from scipy import stats
from matplotlib.pyplot import cm
from functions.get_geometries import get_strike_geometries
from functions.get_perp_pts import perp_pts
from functions.create_shapefiles import create_line_shp
from functions.create_shapefiles import create_polygon_shp
from functions.get_swath_data import get_swath_data

# specify paths
data_path = r"D:/Data/"
results_path = "results/"
shp_path = data_path + r"\HydroATLAS\RiverATLAS_Data_v10_shp\RiverATLAS_v10_shp\RiverATLAS_v10_na.shp"

# check if folder exists
results_path = "results/riveratlas/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

df_tot = gpd.read_file(shp_path)

df = df_tot.loc[df_tot["MAIN_RIV"] == 70765249] #  Colorado 70765249 # Elbe 20282220

print("loading complete")

# area vs discharge
n = 10
bin_edges = stats.mstats.mquantiles(df["ria_ha_usu"], np.linspace(0, 1, n+1))

#mean_stat = stats.binned_statistic(df["ria_ha_usu"], df["dis_m3_pyr"], statistic=lambda y: np.nanmean(y), bins=bin_edges)
#std_stat = stats.binned_statistic(df["ria_ha_usu"], df["dis_m3_pyr"], statistic=lambda y: np.nanstd(y), bins=bin_edges)
median_stat = stats.binned_statistic(df["ria_ha_usu"], df["dis_m3_pyr"], statistic=np.nanmedian, bins=bin_edges)  # bins=nbins, range=bin_range
p_lower_stat = stats.binned_statistic(df["ria_ha_usu"], df["dis_m3_pyr"], statistic=lambda y: np.quantile(y, .25), bins=bin_edges)
p_upper_stat = stats.binned_statistic(df["ria_ha_usu"], df["dis_m3_pyr"], statistic=lambda y: np.quantile(y, .75), bins=bin_edges)
asymmetric_error = [median_stat.statistic - p_lower_stat.statistic,
                     p_upper_stat.statistic - median_stat.statistic]

bin_medians = stats.mstats.mquantiles(df["ria_ha_usu"], np.linspace(0.05, 0.95, n))

f, ax = plt.subplots(figsize=(4, 4))
ax.scatter(df["ria_ha_usu"], df["dis_m3_pyr"], c='grey', s=1, alpha=0.01)#, norm=matplotlib.colors.LogNorm())
ax.errorbar(bin_medians, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
            fmt='o', ms=4, elinewidth=1, c='black', ecolor='black', mec='black', mfc='black', alpha=0.9, label="")
ax.set_xlabel('Upstream area [ha]')
ax.set_ylabel('Discharge [m^3/s]')
#ax.set_xlim([0, 2000])
#ax.set_ylim([0, 1000])
ax.set(xscale='log', yscale='log')
plt.savefig(results_path + "area_vs_discharge" + ".png", dpi=600, bbox_inches='tight')
plt.close()

# area vs discharge with aridity
f, ax = plt.subplots(figsize=(4, 4))
ax.scatter(df["ria_ha_usu"], df["dis_m3_pyr"], c=df["ari_ix_uav"], s=1, alpha=0.01)#, norm=matplotlib.colors.LogNorm())
ax.errorbar(bin_medians, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
            fmt='o', ms=4, elinewidth=1, c='black', ecolor='black', mec='black', mfc='black', alpha=0.9, label="")
ax.set_xlabel('Upstream area [ha]')
ax.set_ylabel('Discharge [m^3/s]')
#ax.set_xlim([0, 2000])
#ax.set_ylim([0, 1000])
ax.set(xscale='log', yscale='log')
plt.savefig(results_path + "area_vs_discharge_aridity" + ".png", dpi=600, bbox_inches='tight')
plt.close()

# area vs slope
n = 10
bin_edges = stats.mstats.mquantiles(df["ria_ha_usu"], np.linspace(0, 1, n+1))

#mean_stat = stats.binned_statistic(df["ria_ha_usu"], df["slp_dg_cav"], statistic=lambda y: np.nanmean(y), bins=bin_edges)
#std_stat = stats.binned_statistic(df["ria_ha_usu"], df["slp_dg_cav"], statistic=lambda y: np.nanstd(y), bins=bin_edges)
median_stat = stats.binned_statistic(df["ria_ha_usu"], df["slp_dg_cav"], statistic=np.nanmedian, bins=bin_edges)  # bins=nbins, range=bin_range
p_lower_stat = stats.binned_statistic(df["ria_ha_usu"], df["slp_dg_cav"], statistic=lambda y: np.quantile(y, .25), bins=bin_edges)
p_upper_stat = stats.binned_statistic(df["ria_ha_usu"], df["slp_dg_cav"], statistic=lambda y: np.quantile(y, .75), bins=bin_edges)
asymmetric_error = [median_stat.statistic - p_lower_stat.statistic,
                     p_upper_stat.statistic - median_stat.statistic]

bin_medians = stats.mstats.mquantiles(df["ria_ha_usu"], np.linspace(0.05, 0.95, n))

f, ax = plt.subplots(figsize=(4, 4))
ax.scatter(df["ria_ha_usu"], df["slp_dg_cav"], c='grey', s=1, alpha=0.01)#, norm=matplotlib.colors.LogNorm())
ax.errorbar(bin_medians, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
            fmt='o', ms=4, elinewidth=1, c='black', ecolor='black', mec='black', mfc='black', alpha=0.9, label="")
ax.set_xlabel('Upstream area [ha]')
ax.set_ylabel('Slope [deg*10]')
#ax.set_xlim([0, 2000])
#ax.set_ylim([0, 1000])
ax.set(xscale='log', yscale='log')
plt.savefig(results_path + "area_vs_slope" + ".png", dpi=600, bbox_inches='tight')
plt.close()

