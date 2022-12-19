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
shp_path = data_path + r"/HydroATLAS/BasinATLAS_Data_v10_shp/BasinATLAS_v10_shp/BasinATLAS_v10_lev09.shp"

# check if folder exists
results_path = "results/HydroAtlas/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

#df_tot = gpd.read_file(shp_path)
df = gpd.read_file(shp_path) # include_fields=["CATCH_SKM", "slp_dg_cav"], ignore_geometry=True

#df = df_tot.loc[df_tot["MAIN_RIV"] == 70765249] #  Colorado 70765249 # Elbe 20282220

print("loading complete")


# area vs slope
n = 10
var2 = "UP_AREA"
var = "sgr_dk_sav"#"slp_dg_sav"
bin_edges = stats.mstats.mquantiles(df[var], np.linspace(0, 1, n+1))

#mean_stat = stats.binned_statistic(df[var], df[var2], statistic=lambda y: np.nanmean(y), bins=bin_edges)
#std_stat = stats.binned_statistic(df[var], df[var2], statistic=lambda y: np.nanstd(y), bins=bin_edges)
median_stat = stats.binned_statistic(df[var], df[var2], statistic=np.nanmedian, bins=bin_edges)  # bins=nbins, range=bin_range
p_lower_stat = stats.binned_statistic(df[var], df[var2], statistic=lambda y: np.quantile(y, .25), bins=bin_edges)
p_upper_stat = stats.binned_statistic(df[var], df[var2], statistic=lambda y: np.quantile(y, .75), bins=bin_edges)
asymmetric_error = [median_stat.statistic - p_lower_stat.statistic,
                     p_upper_stat.statistic - median_stat.statistic]

bin_medians = stats.mstats.mquantiles(df[var], np.linspace(0.05, 0.95, n))

f, ax = plt.subplots(figsize=(4, 4))
ax.scatter(df[var], df[var2], c='grey', s=1, alpha=0.01)#, norm=matplotlib.colors.LogNorm())
ax.errorbar(bin_medians, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
            fmt='o', ms=4, elinewidth=1, c='black', ecolor='black', mec='black', mfc='tab:red', alpha=0.9, label="")
ax.set_xlabel(var)
ax.set_ylabel(var2)
#ax.set_xlim([0, 2000])
ax.set_ylim([10e1, 10e4])
ax.set(xscale='log', yscale='log')
plt.savefig(results_path + var + "_" + var2 + ".png", dpi=600, bbox_inches='tight')
plt.close()

