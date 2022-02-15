import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from scipy import stats

# specify paths
data_path = r"C:/Users/gnann/Documents/QGIS/Topography/"
results_path = r"C:/Users/gnann/Documents/PYTHON/Topography/results/"

shp_path = data_path + "GMBA mountain inventory V1.2(entire world)/GMBA Mountain Inventory_v1.2-World.shp"
dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
clim_path = data_path + "wc2.1_30s_bio/wc2.1_30s_bio_12.tif"
clim_name = "P"

# open raster and plot
dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
clim = rxr.open_rasterio(clim_path, masked=True).squeeze()

"""
f, ax = plt.subplots(figsize=(10, 5))
dem.plot.imshow()
ax.set(title="DEM")
ax.set_axis_off()
plt.show()
"""

# open shapefile and plot
mountain_shp = gpd.read_file(shp_path)

#print('SHP crs: ', mountain_shp.crs)
#print('DEM crs: ', dem.rio.crs)
"""
fig, ax = plt.subplots(figsize=(6, 6))
mountain_shp.plot(ax=ax)
ax.set_title("Shapefile", fontsize=16)
plt.show()
"""

# loop over mountain ranges
mountain_list = ["European Alps"]
    #["Cambrian Mountains", "European Alps", "Pyrenees", "Cordillera Patagonica Sur",
    #             "Ethiopian Highlands", "Himalaya", "Cordillera Central Ecuador", "Sierra Nevada",
    #             "Pennines","Cascade Range", "Appalachian Mountains", "Cordillera Occidental Peru Bolivia Chile"]

for mountain_name in mountain_list:
    mountain_range = mountain_shp.loc[mountain_shp.Name==mountain_name]

    # plot raster and shapefile
    """
    f, ax = plt.subplots(figsize=(10, 5))
    dem.plot.imshow(ax=ax)
    mountain_range.plot(ax=ax, alpha=.8)
    ax.set(title="Raster and shapefile")
    ax.set_axis_off()
    plt.show()
    """

    # clip raster with shapefile
    dem_clipped = dem.rio.clip(mountain_range.geometry.apply(mapping), dem.rio.crs) # This is needed if your GDF is in a diff CRS than the raster data
    clim_clipped = clim.rio.clip(mountain_range.geometry.apply(mapping), dem.rio.crs) # This is needed if your GDF is in a diff CRS than the raster data

    """
    hillshade = es.hillshade(dem_clipped)
    f, ax = plt.subplots(figsize=(10, 4))
    ep.plot_bands(clim_clipped, ax=ax, cmap="Blues")
    #clim_clipped.plot(ax=ax, cmap='Blues', alpha=0.5)
    ax.imshow(hillshade, cmap="Greys", alpha=0.5)
    ax.set(title="Raster clipped using shapefile")
    ax.set_axis_off()
    plt.show()
    """

    # save file
    #dem_clipped.rio.to_raster(results_path + "clipped:dem.tif")

    # calculate elevation profile

    x = clim_clipped.__array__()
    y = dem_clipped.__array__()
    lat = clim_clipped.y.__array__()
    lon = clim_clipped.x.__array__()
    lat = np.vstack([lat]*len(lon)).T

    x = x.flatten()
    y = y.flatten()
    lat = lat.flatten()

    notfinite = (~np.isfinite(x) | ~np.isfinite(x))
    x = np.delete(x, notfinite)
    y = np.delete(y, notfinite)
    lat = np.delete(lat, notfinite)

    bin_edges = stats.mstats.mquantiles(y, np.linspace(0,1,11))
    #mean_stat = stats.binned_statistic(y, x, statistic=lambda y: np.nanmean(y), bins=nbins, range=bin_range)
    # #std_stat = stats.binned_statistic(y, x, statistic=lambda y: np.nanstd(y), bins=nbins, range=bin_range)
    median_stat = stats.binned_statistic(y, x, statistic=np.nanmedian, bins=bin_edges) #bins=nbins, range=bin_range
    p_lower_stat = stats.binned_statistic(y, x, statistic=lambda y: np.quantile(y, .25), bins=bin_edges)
    p_upper_stat = stats.binned_statistic(y, x, statistic=lambda y: np.quantile(y, .75), bins=bin_edges)

    asymmetric_error = [median_stat.statistic - p_lower_stat.statistic, p_upper_stat.statistic - median_stat.statistic]

    #tmp = (median_stat.bin_edges[1:] + median_stat.bin_edges[0:-1]) / 2
    bin_medians = stats.mstats.mquantiles(y, np.linspace(0.05,0.95,10))
    # axs[i].errorbar(tmp, mean_stat.statistic, xerr=None, yerr=std_stat.statistic,
    #                fmt='s', ms=4, elinewidth=1, c=nextcolor, mec=nextcolor, mfc='white', alpha=0.5)

    f, ax = plt.subplots(figsize=(4, 4))
    #ax.plot(x, y, 'o', mfc='none', markersize=.1, alpha=0.1)
    sc = ax.scatter(np.flip(x), np.flip(y), s=0.1, c=lat, alpha=0.2)

    #ax.errorbar(median_stat.statistic, bin_medians, xerr=asymmetric_error, yerr=None,
    #            capsize=2, fmt='s', ms=4, elinewidth=1, c='tab:blue', mfc='white', alpha=0.9)

    ax.set_ylabel('Elevation [m]')
    ax.set_xlabel('Precipitation [mm/year]')
    #ax.set_xlim([0, 5000])
    #ax.set_ylim([0, 6000])
    plt.colorbar(sc)

    plt.savefig(results_path + mountain_name + " " + clim_name + ".png", dpi=600,  bbox_inches='tight')
