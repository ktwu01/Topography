import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio
import plotting_fcts

# specify paths
data_path = "/home/hydrosys/data/" #r"C:/Users/Sebastian/Documents/Data/"
#data_path = r"D:/Data/"
results_path = "results/"

gdf = gpd.read_file(results_path + 'dataframe.shp')
#gdf = gdf.dropna()
gdf.loc[gdf["aridity"] == np.inf, "aridity"] = np.nan

var = "conv"
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()#projection=ccrs.Robinson()
gdf.loc[gdf["aridity"] < 1, "wtd"]
ax.scatter(gdf.loc[gdf["aridity"] > 1, var], gdf.loc[gdf["aridity"] > 1, "wtd"], s=1, facecolor='tab:orange', edgecolor='none', alpha=0.1)
plotting_fcts.plot_bins(gdf.loc[gdf["aridity"] > 1, var], gdf.loc[gdf["aridity"] > 1, "wtd"], fillcolor='tab:orange')
ax.scatter(gdf.loc[gdf["aridity"] < 1, var], gdf.loc[gdf["aridity"] < 1, "wtd"], s=1, facecolor='tab:blue', edgecolor='none', alpha=0.1)
plotting_fcts.plot_bins(gdf.loc[gdf["aridity"] < 1, var], gdf.loc[gdf["aridity"] < 1, "wtd"], fillcolor='tab:blue')
#ax.set_xlabel('Slope')
ax.set_xlabel(var)
ax.set_ylabel('WTD [m]')
#ax.set_xlim([0, 1000])
#ax.set_xscale('log')
#ax.set_xlim([-100, 3900])
ax.set_ylim([.1, 100])
ax.set_yscale('log')
#idx = np.isfinite(gdf['aridity']) & np.isfinite(gdf['wtd'])
#m, b = np.polyfit(gdf['aridity'][idx], gdf['wtd'][idx], 1)
#plt.plot(np.linspace(0,10000,10), m*np.linspace(0,10000,10) + b)
#plt.show()
#print(stats.spearmanr(gdf['value'], gdf['wtd'], nan_policy='omit'))
plt.savefig(results_path + "wtd_vs_" + var + ".png", dpi=600, bbox_inches='tight')
plt.close()

