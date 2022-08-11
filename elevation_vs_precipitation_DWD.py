import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
import matplotlib
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# This script ...

# prepare data
data_path = "data/"

# check if folder exists
results_path = "figures/dwd/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
df = pd.read_csv(data_path + "Niederschlag_1991-2020.txt", sep=';' , encoding='latin-1')
df_tmp = pd.read_csv(data_path + "Niederschlag_1991-2020_Stationsliste.txt", sep=';' , encoding='latin-1')

df = pd.merge(df, df_tmp, on=['Stations_id'], how='outer')
df = df[["Jahr", "Stationshoehe", "geogr. Laenge"]].dropna()

r_sp1, _ = stats.spearmanr(df["Jahr"], df["Stationshoehe"], nan_policy='omit')

n = 10
bin_edges = stats.mstats.mquantiles(df["Stationshoehe"], np.linspace(0, 1, n+1))

mean_stat1 = stats.binned_statistic(df["Stationshoehe"], df["Jahr"], statistic=lambda y: np.nanmean(y), bins=bin_edges)
std_stat1 = stats.binned_statistic(df["Stationshoehe"], df["Jahr"], statistic=lambda y: np.nanstd(y), bins=bin_edges)
median_stat1 = stats.binned_statistic(df["Stationshoehe"], df["Jahr"], statistic=np.nanmedian, bins=bin_edges)  # bins=nbins, range=bin_range
p_lower_stat1 = stats.binned_statistic(df["Stationshoehe"], df["Jahr"], statistic=lambda y: np.quantile(y, .25), bins=bin_edges)
p_upper_stat1 = stats.binned_statistic(df["Stationshoehe"], df["Jahr"], statistic=lambda y: np.quantile(y, .75), bins=bin_edges)
asymmetric_error1 = [median_stat1.statistic - p_lower_stat1.statistic,
                     p_upper_stat1.statistic - median_stat1.statistic]

bin_medians = stats.mstats.mquantiles(df["Stationshoehe"], np.linspace(0.05, 0.95, n))

# plot precipitation
f, ax = plt.subplots(figsize=(4, 4))
ax.scatter(df["Jahr"],df["Stationshoehe"], c=df["geogr. Laenge"], s=1, alpha=0.5)#, norm=matplotlib.colors.LogNorm())
#ax.scatter(mean_stat1.statistic, bin_medians, s=5, c='black', label='Precipitation')
#ax.plot(mean_stat1.statistic, bin_medians, c='tab:blue', label='Precipitation')
#ax.fill_betweenx(bin_medians, mean_stat1.statistic - std_stat1.statistic, mean_stat1.statistic + std_stat1.statistic,
#                 facecolor='tab:blue', alpha=0.25)
ax.annotate(r' $\rho_s$ ' "= " + str(np.round(r_sp1,2)) + "\n", xy=(.09, .7), xycoords=ax.transAxes, fontsize=10)
ax.set_ylabel('Elevation [m]')
ax.set_xlabel('P [mm/year]')
ax.set_xlim([0, 2000])
ax.set_ylim([0, 1000])
#ax.set(xscale='linear', yscale='log')
plt.savefig(results_path + "dwd_pr_vs_elevation" + ".png", dpi=600, bbox_inches='tight')
plt.close()