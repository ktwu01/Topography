import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import geopandas as gpd
from scipy import stats

#todo: clean up a bit...

# specify paths
data_path = r"C:/Users/Sebastian/Documents/Data/"
#data_path = r"D:/Data/"
results_path = "results/"

shp_path = data_path + "GMBA mountain inventory V1.2(entire world)/GMBA Mountain Inventory_v1.2-World.shp"
dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
pr_path = data_path + "wc2.1_30s_bio/wc2.1_30s_bio_12.tif"
pet_path = data_path + "7504448/global-et0_annual.tif/et0_yr/et0_yr.tif"
t_path = data_path + "wc2.1_30s_bio/wc2.1_30s_bio_1.tif"
p_name = "P"
pet_name = "PET"
t_name = "T"

# open raster and plot
dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
pr = rxr.open_rasterio(pr_path, masked=True).squeeze()
pet = rxr.open_rasterio(pet_path, masked=True).squeeze()
t = rxr.open_rasterio(t_path, masked=True).squeeze()

"""
f, ax = plt.subplots(figsize=(10, 5))
dem.plot.imshow()
ax.set(title="DEM")
ax.set_axis_off()
plt.show()
"""

# open shapefile and plot
mountain_shp = gpd.read_file(shp_path)

"""
fig, ax = plt.subplots(figsize=(6, 6))
mountain_shp.plot(ax=ax)
ax.set_title("Shapefile", fontsize=16)
plt.show()
"""

# loop over mountain ranges
mountain_list = ["Cambrian Mountains", "European Alps", "Pyrenees", "Cordillera Patagonica Sur",
                 "Ethiopian Highlands", "Himalaya", "Cordillera Central Ecuador", "Sierra Nevada",
                 "Pennines", "Cascade Range", "Appalachian Mountains", "Cordillera Occidental Peru Bolivia Chile"]
mountain_list = ["Cordillera Central Ecuador", "Tenerife", "Himalaya", "European Alps", "Scandinavian Mountains"]

for mountain_name in mountain_list:

    mountain_range = mountain_shp.loc[mountain_shp.Name == mountain_name]

    # check if folder exists
    path = results_path + mountain_name + "/"
    if not os.path.isdir(path):
        os.makedirs(path)

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
    dem_clipped = dem.rio.clip(mountain_range.geometry.apply(mapping), dem.rio.crs)
    pr_clipped = pr.rio.clip(mountain_range.geometry.apply(mapping), dem.rio.crs)
    pet_clipped = pet.rio.clip(mountain_range.geometry.apply(mapping), dem.rio.crs)
    t_clipped = t.rio.clip(mountain_range.geometry.apply(mapping), dem.rio.crs)

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

    x1 = pr_clipped.__array__()
    x2 = pet_clipped.__array__()
    x3 = t_clipped.__array__()*100
    y = dem_clipped.__array__()
    lat = dem_clipped.y.__array__()
    lon = dem_clipped.x.__array__()
    lat = np.vstack([lat]*len(lon)).T
    lon = np.vstack([lon]*len(lat))

    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    y = y.flatten()
    lat = lat.flatten()
    lon = lon.flatten()

    notfinite = (~np.isfinite(x1) | ~np.isfinite(x2) | ~np.isfinite(x3) | ~np.isfinite(y))
    x1 = np.delete(x1, notfinite)
    x2 = np.delete(x2, notfinite)
    x3 = np.delete(x3, notfinite)
    y = np.delete(y, notfinite)
    lat = np.delete(lat, notfinite)
    lon = np.delete(lon, notfinite)

    plot_idx = np.random.permutation(x1.shape[0])

    bin_edges = stats.mstats.mquantiles(y, np.linspace(0,1,51))

    mean_stat1 = stats.binned_statistic(y, x1, statistic=lambda y: np.nanmean(y), bins=bin_edges)
    std_stat1 = stats.binned_statistic(y, x1, statistic=lambda y: np.nanstd(y), bins=bin_edges)
    median_stat1 = stats.binned_statistic(y, x1, statistic=np.nanmedian, bins=bin_edges) #bins=nbins, range=bin_range
    p_lower_stat1 = stats.binned_statistic(y, x1, statistic=lambda y: np.quantile(y, .25), bins=bin_edges)
    p_upper_stat1 = stats.binned_statistic(y, x1, statistic=lambda y: np.quantile(y, .75), bins=bin_edges)
    asymmetric_error1 = [median_stat1.statistic - p_lower_stat1.statistic,
                         p_upper_stat1.statistic - median_stat1.statistic]

    mean_stat2 = stats.binned_statistic(y, x2, statistic=lambda y: np.nanmean(y), bins=bin_edges)
    std_stat2 = stats.binned_statistic(y, x2, statistic=lambda y: np.nanstd(y), bins=bin_edges)
    median_stat2 = stats.binned_statistic(y, x2, statistic=np.nanmedian, bins=bin_edges) #bins=nbins, range=bin_range
    p_lower_stat2 = stats.binned_statistic(y, x2, statistic=lambda y: np.quantile(y, .25), bins=bin_edges)
    p_upper_stat2 = stats.binned_statistic(y, x2, statistic=lambda y: np.quantile(y, .75), bins=bin_edges)
    asymmetric_error2 = [median_stat2.statistic - p_lower_stat2.statistic,
                         p_upper_stat2.statistic - median_stat2.statistic]

    mean_stat3 = stats.binned_statistic(y, x3, statistic=lambda y: np.nanmean(y), bins=bin_edges)
    std_stat3 = stats.binned_statistic(y, x3, statistic=lambda y: np.nanstd(y), bins=bin_edges)
    median_stat3 = stats.binned_statistic(y, x3, statistic=np.nanmedian, bins=bin_edges)  # bins=nbins, range=bin_range
    p_lower_stat3 = stats.binned_statistic(y, x3, statistic=lambda y: np.quantile(y, .25), bins=bin_edges)
    p_upper_stat3 = stats.binned_statistic(y, x3, statistic=lambda y: np.quantile(y, .75), bins=bin_edges)
    asymmetric_error3 = [median_stat3.statistic - p_lower_stat3.statistic,
                         p_upper_stat3.statistic - median_stat3.statistic]

    #bin_means = (mean_stat1.bin_edges[1:] + mean_stat1.bin_edges[0:-1]) / 2
    bin_medians = stats.mstats.mquantiles(y, np.linspace(0.05,0.95,50))

    f, ax = plt.subplots(figsize=(4, 4))
    """
    #ax.plot(x, y, 'o', mfc='none', markersize=.1, alpha=0.1)
    sc = ax.scatter(x1, y, s=0.025, c='tab:blue', alpha=0.05)
    sc = ax.scatter(x2, y, s=0.025, c='tab:orange', alpha=0.05)
    ax.errorbar(median_stat1.statistic, bin_medians, xerr=asymmetric_error1, yerr=None,
                capsize=2, fmt='s', ms=4, elinewidth=1, c='tab:blue', mfc='white', alpha=0.7)
    ax.errorbar(median_stat2.statistic, bin_medians, xerr=asymmetric_error2, yerr=None,
                capsize=2, fmt='s', ms=4, elinewidth=1, c='tab:orange', mfc='white', alpha=0.7)
    """
    ax.plot(mean_stat1.statistic, bin_medians, c='tab:blue', label='Precipitation')
    ax.fill_betweenx(bin_medians, mean_stat1.statistic - std_stat1.statistic, mean_stat1.statistic + std_stat1.statistic,
                     facecolor='tab:blue', alpha=0.25)

    ax.plot(mean_stat2.statistic, bin_medians, c='tab:orange', label='Potential evapotranspiration')
    ax.fill_betweenx(bin_medians, mean_stat2.statistic - std_stat2.statistic, mean_stat2.statistic + std_stat2.statistic,
                     facecolor='tab:orange', alpha=0.25)

    #ax.plot(mean_stat3.statistic, bin_medians, c='tab:purple', label='Temperature')
    #ax.fill_betweenx(bin_medians, mean_stat3.statistic - std_stat3.statistic,
    #                 mean_stat3.statistic + std_stat3.statistic,
    #                 facecolor='tab:purple', alpha=0.25)

    ax.set_ylabel('Elevation [m]')
    ax.set_xlabel('P or PET [mm/year]') # or T [100*°C]
    #ax.set_xlim([0, 5000])
    #ax.set_ylim([0, 6000])
    #plt.colorbar(sc)

    #plt.show()
    plt.savefig(results_path + mountain_name + "/" + "climatic_water_balance" + ".png", dpi=600,  bbox_inches='tight')
    plt.close()