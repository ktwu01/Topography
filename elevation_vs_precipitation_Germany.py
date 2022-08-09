import os
import matplotlib.pyplot as plt
import matplotlib.colors as ml_colors
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import geopandas as gpd
import pandas as pd
from brewer2mpl import brewer2mpl
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from matplotlib.pyplot import cm
#import pyosp
import fiona
from functions.get_geometries import get_strike_geometries
from functions.get_perp_pts import perp_pts
from functions.create_shapefiles import create_line_shp
from functions.create_shapefiles import create_polygon_shp
from functions.get_swath_data import get_swath_data
from functions.get_geometries import get_swath_indices
from scipy import stats

# specify paths
data_path = r"C:/Users/Sebastian/Documents/Data/"
data_path = r"D:/Data/"
results_path = "results/"

shp_path = data_path + "GMBA mountain inventory V1.2(entire world)/GMBA Mountain Inventory_v1.2-World.shp"
dem_path = data_path + "CHELSA/CHELSA_hyd_glo_dem_30s.tif"
#dem_path = data_path + "mn30_grd/mn30_grd/w001001.adf"
pr_path = data_path + "CHELSA/CHELSA_bio12_1981-2010_V.2.1.tif"
pet_path = data_path + "CHELSA/CHELSA_pet_penman_mean_1981-2010_V.2.1.tif"
t_path = data_path + "CHELSA/CHELSA_bio1_1981-2010_V.2.1.tif"
#dem_path = data_path + "hyd_glo_dem_30s/hyd_glo_dem_30s.tif"

dem_path = data_path + "WorldClim/wc2.1_30s_elev/wc2.1_30s_elev.tif"
pr_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_12.tif"
pet_path = data_path + "WorldClim/7504448/global-et0_annual.tif/et0_yr/et0_yr.tif"
#vap_path = data_path + "WorldClim/wc2.1_30s_vapr/wc2.1_30s_vapr_avg.tif"
t_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_1.tif"

germany_path = data_path + "DEU_adm/DEU_adm0.shp"


# check if folders exist
path = results_path + "Germany/"
if not os.path.isdir(path):
    os.makedirs(path)

germany = gpd.read_file(germany_path)

"""
# load files
dem = rxr.open_rasterio(dem_path, masked=True).squeeze() #todo: remove masked ...
pr = rxr.open_rasterio(pr_path, masked=True).squeeze()
pet = rxr.open_rasterio(pet_path, masked=True).squeeze()
t = rxr.open_rasterio(t_path, masked=True).squeeze()

# clip grids using shapefile
dem_clipped = dem.rio.clip(germany.geometry.apply(mapping), dem.rio.crs)
pr_clipped = pr.rio.clip(germany.geometry.apply(mapping), dem.rio.crs)
pet_clipped = pet.rio.clip(germany.geometry.apply(mapping), dem.rio.crs)
t_clipped = t.rio.clip(germany.geometry.apply(mapping), dem.rio.crs)
"""

var_list = ["pr", "pet", "t", "elevation"]
path_list = [pr_path, pet_path, t_path, dem_path]
#data_list = []

# open all datasets
df = pd.DataFrame(columns=["y", "x"])
for var, path in zip(var_list, path_list):
    rds = rxr.open_rasterio(path)
    rds = rds.rio.clip(germany.geometry.apply(mapping), rds.rio.crs)
    #data_list.append(rds)
    rds = rds.squeeze().drop("spatial_ref").drop("band")
    rds.name = var
    df_tmp = rds.to_dataframe().reset_index()
    df_tmp["y"] = np.round(df_tmp["y"],4) # because of small coordinate differences...
    df_tmp["x"] = np.round(df_tmp["x"],4)
    df = pd.merge(df, df_tmp, on=['y', 'x'], how='outer')

df.rename(columns={'x': 'lon', 'y': 'lat'}, inplace=True)
#df["pr"] = df["pr"] * 0.1
#df["pet"] = df["pet"] * 0.1
#df["t"] = df["t"] * 0.1 - 273.15
df['aridity'] = df['pet'] / df['pr']

# remove values outside of continents
df.loc[df["pet"] > 5000] = np.nan
# df.loc[df["pr"] > 10000] = np.nan
# df.loc[df["t"] < 30] = np.nan
df.loc[df["elevation"] < -10] = np.nan
df.loc[df["elevation"] > 10000] = np.nan

df = df.dropna()

# correlations
r_sp1, _ = stats.spearmanr(df["pr"], df["elevation"], nan_policy='omit')
r_sp2, _ = stats.spearmanr(df["pet"], df["elevation"], nan_policy='omit')
r_sp3, _ = stats.spearmanr(df["t"], df["elevation"], nan_policy='omit')

# bins
n = 50
bin_edges = stats.mstats.mquantiles(df["elevation"], np.linspace(0, 1, n+1))

mean_stat1 = stats.binned_statistic(df["elevation"], df["pr"], statistic=lambda y: np.nanmean(y), bins=bin_edges)
std_stat1 = stats.binned_statistic(df["elevation"], df["pr"], statistic=lambda y: np.nanstd(y), bins=bin_edges)
median_stat1 = stats.binned_statistic(df["elevation"], df["pr"], statistic=np.nanmedian, bins=bin_edges)  # bins=nbins, range=bin_range
p_lower_stat1 = stats.binned_statistic(df["elevation"], df["pr"], statistic=lambda y: np.quantile(y, .25), bins=bin_edges)
p_upper_stat1 = stats.binned_statistic(df["elevation"], df["pr"], statistic=lambda y: np.quantile(y, .75), bins=bin_edges)
asymmetric_error1 = [median_stat1.statistic - p_lower_stat1.statistic,
                     p_upper_stat1.statistic - median_stat1.statistic]

mean_stat2 = stats.binned_statistic(df["elevation"], df["pet"], statistic=lambda y: np.nanmean(y), bins=bin_edges)
std_stat2 = stats.binned_statistic(df["elevation"], df["pet"], statistic=lambda y: np.nanstd(y), bins=bin_edges)
median_stat2 = stats.binned_statistic(df["elevation"], df["pet"], statistic=np.nanmedian, bins=bin_edges)  # bins=nbins, range=bin_range
p_lower_stat2 = stats.binned_statistic(df["elevation"], df["pet"], statistic=lambda y: np.quantile(y, .25), bins=bin_edges)
p_upper_stat2 = stats.binned_statistic(df["elevation"], df["pet"], statistic=lambda y: np.quantile(y, .75), bins=bin_edges)
asymmetric_error2 = [median_stat2.statistic - p_lower_stat2.statistic,
                     p_upper_stat2.statistic - median_stat2.statistic]


# bin_means = (mean_stat1.bin_edges[1:] + mean_stat1.bin_edges[0:-1]) / 2
bin_medians = stats.mstats.mquantiles(df["elevation"], np.linspace(0.05, 0.95, n))

# plot precipitation
f, ax = plt.subplots(figsize=(4, 4))
ax.scatter(df["pr"],df["elevation"], c="grey", s=0.05, alpha=0.1)
ax.scatter(mean_stat1.statistic, bin_medians, s=5, c='tab:blue', label='Precipitation')
#ax.plot(mean_stat1.statistic, bin_medians, c='tab:blue', label='Precipitation')
#ax.fill_betweenx(bin_medians, mean_stat1.statistic - std_stat1.statistic, mean_stat1.statistic + std_stat1.statistic,
#                 facecolor='tab:blue', alpha=0.25)
ax.annotate(r' $\rho_s$ ' "= " + str(np.round(r_sp1,2)) + "\n", xy=(.09, .7), xycoords=ax.transAxes, fontsize=10)
ax.set_ylabel('Elevation [m]')
ax.set_xlabel('P [mm/year]')
#ax.set(xscale='log', yscale='log')
plt.savefig(results_path + "germany" + "/" + "pr" + ".png", dpi=600, bbox_inches='tight')
plt.close()

# plot potential evapotranspiration
f, ax = plt.subplots(figsize=(4, 4))
ax.scatter(df["pet"],df["elevation"], c="grey", s=0.05, alpha=0.1)
ax.scatter(mean_stat2.statistic, bin_medians, s=5, c='tab:orange', label='Potential Evapotranspiration')
#ax.plot(mean_stat2.statistic, bin_medians, c='tab:orange', label='Potential Evapotranspiration')
#ax.fill_betweenx(bin_medians, mean_stat2.statistic - std_stat2.statistic, mean_stat2.statistic + std_stat2.statistic,
#                 facecolor='tab:orange', alpha=0.25)
ax.annotate(r' $\rho_s$ ' "= " + str(np.round(r_sp2,2)) + "\n", xy=(.09, .7), xycoords=ax.transAxes, fontsize=10)
ax.set_ylabel('Elevation [m]')
ax.set_xlabel('PET [mm/year]')

plt.savefig(results_path + "germany" + "/" + "pet" + ".png", dpi=600, bbox_inches='tight')
plt.close()

# plot map
for var in ["elevation", "pr", "pet"]:
    bounds=np.linspace(0, 1000, 11)
    colormap='YlGnBu'
    o = brewer2mpl.get_map(colormap, 'Sequential', 9, reverse=False)
    c = o.mpl_colormap

    # create figure
    plt.rcParams['axes.linewidth'] = 0.1
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    customnorm = ml_colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    sc = ax.scatter(df["lon"], df["lat"], c=df[var], cmap=c, marker='s', s=.35, edgecolors='none',
                    norm=customnorm, transform=ccrs.PlateCarree())
    # ax.coastlines(linewidth=0.5)

    box = sgeom.box(minx=5, maxx=15, miny=47, maxy=55)
    x0, y0, x1, y1 = box.bounds
    ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())

    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.5, extend='max')
    cbar.set_label(var)
    # cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
    cbar.ax.tick_params(labelsize=12)
    plt.gca().outline_patch.set_visible(False)

    plt.savefig(results_path + "germany" + "/" + var + "_map.png", dpi=600, bbox_inches='tight')
    plt.close()

