import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio
import plotting_fcts
from scipy import stats

# specify paths
data_path = "/home/hydrosys/data/" #r"C:/Users/Sebastian/Documents/Data/"
#data_path = r"D:/Data/"
results_path = "results/"

gdf = gpd.read_file(results_path + 'dataframe.shp')
gdf.loc[gdf["aridity"] == np.inf, "aridity"] = np.nan
gdf = gdf.dropna()

# todo: make bin work for geom
# todo: clean up if statements...

var_list = ["dem", "slope", "conv", "aridity", "twi", "geom"]
for var in var_list:
    print(var)
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()#projection=ccrs.Robinson()
    gdf.loc[gdf["aridity"] < 1, "wtd"]
    ax.scatter(gdf.loc[gdf["aridity"] > 1, var], gdf.loc[gdf["aridity"] > 1, "wtd"], s=1, facecolor='tab:orange', edgecolor='none', alpha=0.1)
    plotting_fcts.plot_bins(gdf.loc[gdf["aridity"] > 1, var], gdf.loc[gdf["aridity"] > 1, "wtd"], fillcolor='tab:orange')
    ax.scatter(gdf.loc[gdf["aridity"] < 1, var], gdf.loc[gdf["aridity"] < 1, "wtd"], s=1, facecolor='tab:blue', edgecolor='none', alpha=0.1)
    plotting_fcts.plot_bins(gdf.loc[gdf["aridity"] < 1, var], gdf.loc[gdf["aridity"] < 1, "wtd"], fillcolor='tab:blue')
    ax.set_xlabel(var)
    ax.set_ylabel('WTD [m]')
    if var == "conv":
        ax.set_xlim([np.nanquantile(gdf[var],0.01), np.nanquantile(gdf[var],0.99)])
    elif var == "geom":
        pass
    elif var == "twi":
        ax.set_xlim([np.nanquantile(gdf[var],0.01), np.nanquantile(gdf[var],0.99)])
    else:
        ax.set_xscale('log')
        ax.set_xlim([np.nanquantile(gdf[var],0.01), np.nanquantile(gdf[var],0.99)])
    #ax.set_ylim([np.nanquantile(gdf["wtd"],0.01), np.nanquantile(gdf["wtd"],0.99)])
    ax.set_ylim([.1, 100])
    ax.set_yscale('log')
    #if var != "aridity":
    rho_s1, _ = stats.spearmanr(gdf.loc[gdf["aridity"] > 1, var], gdf.loc[gdf["aridity"] > 1, "wtd"], nan_policy='omit')
    rho_s2, _ = stats.spearmanr(gdf.loc[gdf["aridity"] < 1, var], gdf.loc[gdf["aridity"] < 1, "wtd"])
    ax.annotate("rho_s arid: {:.2f} ".format(rho_s1), xy=(.1, .9), xycoords=ax.transAxes, fontsize=10)
    ax.annotate("rho_s humid: {:.2f} ".format(rho_s2), xy=(.1, .85), xycoords=ax.transAxes, fontsize=10)
    plt.savefig(results_path + "wtd_vs_" + var + ".png", dpi=600, bbox_inches='tight')
    plt.close()

