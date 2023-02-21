import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
import seaborn as sns
from functions import plotting_fcts
from functions import get_nearest_neighbour
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio

# todo: use other P data than ISIMIP

# This script ...

# prepare data
data_path = "D:/Data/"
"""
slope_path = "D:/Data/DEMs/Geomorpho90m/" + "dtm_slope_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
twi_path = "D:/Data/DEMs/Geomorpho90m/" + "dtm_cti_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
#twi_path = "C:/Users/gnann/Desktop/dtm_twi_merit.dem_m_1km_s0..0cm_2017_v1.0.tif"
elevation_path = "D:/Data/DEMs/MERIT_250m/" + "dtm_elevation_merit.dem_m_250m_s0..0cm_2017_v1.0.tif"
pr_path = "D:/Data/CHELSA/CHELSA_bio12_1981-2010_V.2.1.tif"
pet_path = "D:/Data/CHELSA/CHELSA_pet_penman_mean_1981-2010_V.2.1.tif"
"""

pr_path = data_path + "resampling/" + "P_CHELSA_30s.tif"
pet_path = data_path + "resampling/" + "PET_CHELSA_30s.tif"
slope_path = data_path + "resampling/" + "Slope_MERIT_30s.tif"
elevation_path = data_path + "resampling/" + "Elevation_MERIT_30s.tif"
landform_path = data_path + "resampling/" + "WorldLandform_30sec.tif"

slope = rio.open(slope_path, masked=True)
elevation = rio.open(elevation_path, masked=True)
pr = rio.open(pr_path, masked=True)
pet = rio.open(pet_path, masked=True)
landform = rio.open(landform_path, masked=True)

caravan_path = "data/complete_table.csv" #"D:/Data/Caravan/complete_table.csv" #
#data_path = "data/attributes/camels/attributes_hydroatlas_camels.csv"
#data_path = "data/attributes/hysets/attributes_hydroatlas_hysets.csv"

# check if folder exists
results_path = "results/Caravan/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
df = pd.read_csv(caravan_path, sep=',')

#df_domains = pd.read_csv("./2b/aggregated/domains.csv", sep=',')
#geometry = [Point(xy) for xy in zip(df_domains.lon, df_domains.lat)]
#gdf_domains = gpd.GeoDataFrame(df_domains, geometry=geometry)

#closest = get_nearest_neighbour.nearest_neighbor(gdf, gdf_domains, return_dist=True)
#closest = get_nearest_neighbour.nearest_neighbor(gdf_domains, gdf, return_dist=True)
#closest = closest.rename(columns={'geometry': 'closest_geom'})
# merge the datasets by index (for this, it is good to use '.join()' -function)
#df = gdf.join(closest)

coord_list = [(x, y) for x, y in zip(df['gauge_lon'], df['gauge_lat'])]

df['slope_30s'] = [x for x in slope.sample(coord_list)]
df['slope_30s'] = np.concatenate(df['slope_30s'].to_numpy())
df.loc[df["slope_30s"] < 0, "slope_30s"] = np.nan
df['slope_30s'] = np.tan(np.deg2rad(df['slope_30s'] * 0.01))

df['elevation_30s'] = [x for x in elevation.sample(coord_list)]
df['elevation_30s'] = np.concatenate(df['elevation_30s'].to_numpy())
df.loc[df["elevation_30s"] < -1000, "elevation_30s"] = np.nan

df['pr_30s'] = [x for x in pr.sample(coord_list)]
df['pr_30s'] = np.concatenate(df['pr_30s'].to_numpy())
df.loc[df["pr_30s"] > 50000, "pr_30s"] = np.nan
df['pr_30s'] = df['pr_30s'] * 0.1

df['pet_30s'] = [x for x in pet.sample(coord_list)]
df['pet_30s'] = np.concatenate(df['pet_30s'].to_numpy())
df.loc[df["pet_30s"] > 50000, "pet_30s"] = np.nan
df['pet_30s'] = df['pet_30s'] * 0.01 * 12

df['landform'] = [x for x in landform.sample(coord_list)]
df['landform'] = np.concatenate(df['landform'].to_numpy())
df.loc[df["landform"] < 1, "landform"] = np.nan
df.loc[df["landform"] > 4, "landform"] = np.nan

df["aridity_30s"] = df["pet_30s"]/df["pr_30s"]

df["dummy"] = ""

#df = df.dropna()


x_name = "aridity_30s"
y_name = "elevation_30s"
x_unit = " [-]"
y_unit = " [m]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=0.1, s=10, lw = 0, color='black')
g.set(xlim=[0.1, 100], ylim=[-10, 5000])
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_distribution.png", dpi=600, bbox_inches='tight')
plt.close()

df.loc[df["landform"]==1, "landform"] = 5 # mountains
df.loc[df["landform"]==2, "landform"] = 5 # hills
df.loc[df["landform"]==3, "landform"] = 5 # plateaus
df.loc[df["landform"]==4, "landform"] = 6 # plains

x_name = "aridity_30s"
y_name = "slope_30s"
x_unit = " [-]"
y_unit = " [m]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", hue="landform", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=0.2, s=5, lw=0)
g.set(xlim=[0.2, 20], ylim=[0.001, 1])
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='log')
#plt.legend(loc='upper right')
plt.savefig(results_path + x_name + '_' + y_name + "_distribution.png", dpi=600, bbox_inches='tight')
plt.close()

# count above 1000m (or mountains and hills) and humid
threshold = 0.08
df_tmp = df["slope_30s"]
print("Distribution topography")
print("Arid and fraction below " + str(threshold) + ": " + str(
    round(len(df_tmp[np.logical_and(df_tmp <= threshold,df["aridity_30s"]>1)]) / len(df_tmp), 2)))
print("Humid and fraction below " + str(threshold) + ": " + str(
    round(len(df_tmp[np.logical_and(df_tmp <= threshold, df["aridity_30s"]<1)]) / len(df_tmp), 2)))
print("Arid and fraction above " + str(threshold) + ": " + str(
    round(len(df_tmp[np.logical_and(df_tmp > threshold, df["aridity_30s"]>1)]) / len(df_tmp), 2)))
print("Humid and fraction above " + str(threshold) + ": " + str(
    round(len(df_tmp[np.logical_and(df_tmp > threshold,df["aridity_30s"]<1)]) / len(df_tmp), 2)))

df_tmp = df["landform"]
print("Distribution landforms")
print("Humid and plains " + ": " + str(
    round(len(df_tmp[np.logical_and(df["landform"]==6, df["aridity_30s"]<1)]) / len(df_tmp), 2)))
print("Arid and plains " + ": " + str(
    round(len(df_tmp[np.logical_and(df["landform"]==6, df["aridity_30s"]>1)]) / len(df_tmp), 2)))
print("Humid and mountains " + ": " + str(
    round(len(df_tmp[np.logical_and(df["landform"]==5, df["aridity_30s"]<1)]) / len(df_tmp), 2)))
print("Arid and mountains " + ": " + str(
    round(len(df_tmp[np.logical_and(df["landform"]==5, df["aridity_30s"]>1)]) / len(df_tmp), 2)))


"""
############
#
x_name = "Aridity"
y_name = "Slope"
x_unit = " [-]"
y_unit = " [m]"
#df["aridity_class"] = "energy-limited"
#df.loc[df["Aridity"] > 1, "aridity_class"] = "water-limited"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
#g.map_dataframe(plotting_fcts.plot_coloured_scatter_random, x_name, y_name,
#                domains="aridity_class", alpha=0.1, s=10)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw=0, color='lightgrey')
g.set(xlim=[0.2, 20], ylim=[1, 10000])
#g.map(plotting_fcts.plot_origin_line, var, wtd)
#g.map(sns.rugplot, var, wtd, lw=1, alpha=.002, color="lightgrey")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='log')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

#
x_name = "Aridity"
y_name = "Slope"
x_unit = " [-]"
y_unit = " [m]"
#df["aridity_class"] = "energy-limited"
#df.loc[df["Aridity"] > 1, "aridity_class"] = "water-limited"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
#g.map_dataframe(plotting_fcts.plot_coloured_scatter_random, x_name, y_name,
#                domains="aridity_class", alpha=0.1, s=10)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw=0, color='lightgrey')
g.set(xlim=[0.2, 20], ylim=[0.0001, 1])
#g.map(plotting_fcts.plot_origin_line, var, wtd)
#g.map(sns.rugplot, var, wtd, lw=1, alpha=.002, color="lightgrey")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='log')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

#
x_name = "Elevation"
y_name = "Recharge"
x_unit = " [m]"
y_unit = " [mm/y]"
#df["aridity_class"] = "energy-limited"
#df.loc[df["Aridity"] > 1, "aridity_class"] = "water-limited"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
#g.map_dataframe(plotting_fcts.plot_coloured_scatter_random, x_name, y_name,
#                domains="aridity_class", alpha=0.1, s=10)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw=0, color='lightgrey')
g.set(xlim=[1, 10000], ylim=[-100, 1400])
#g.map(plotting_fcts.plot_origin_line, var, wtd)
#g.map(sns.rugplot, var, wtd, lw=1, alpha=.002, color="lightgrey")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
#g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

#
x_name = "Slope"
y_name = "Recharge"
x_unit = " [-]"
y_unit = " [mm/y]"
#df["aridity_class"] = "energy-limited"
#df.loc[df["Aridity"] > 1, "aridity_class"] = "water-limited"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
#g.map_dataframe(plotting_fcts.plot_coloured_scatter_random, x_name, y_name,
#                domains="aridity_class", alpha=0.1, s=10)
g.set(xlim=[0.001, 1], ylim=[-100, 2000])
#g.map(plotting_fcts.plot_origin_line, var, wtd)
#g.map(sns.rugplot, var, wtd, lw=1, alpha=.002, color="lightgrey")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:green", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

#
x_name = "Elevation"
y_name = "Precipitation"
x_unit = " [-]"
y_unit = " [mm/y]"
#df["aridity_class"] = "energy-limited"
#df.loc[df["Aridity"] > 1, "aridity_class"] = "water-limited"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
#g.map_dataframe(plotting_fcts.plot_coloured_scatter_random, x_name, y_name,
#                domains="aridity_class", alpha=0.1, s=10)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
g.set(xlim=[1, 10000], ylim=[-100, 3000])
#g.map(plotting_fcts.plot_origin_line, var, wtd)
#g.map(sns.rugplot, var, wtd, lw=1, alpha=.002, color="lightgrey")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()


x_name = "Precipitation"
y_name = "Recharge"
x_unit = " [mm/y]"
y_unit = " [mm/y]"
#df["aridity_class"] = "energy-limited"
#df.loc[df["Aridity"] > 1, "aridity_class"] = "water-limited"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
#g.map_dataframe(plotting_fcts.plot_coloured_scatter_random, x_name, y_name,
#                domains="aridity_class", alpha=0.1, s=10)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
g.set(xlim=[0, 3000], ylim=[0, 2000])
#g.map(plotting_fcts.plot_origin_line, var, wtd)
#g.map(sns.rugplot, var, wtd, lw=1, alpha=.002, color="lightgrey")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='linear', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Aridity"
y_name = "Recharge"
x_unit = " [-]"
y_unit = " [mm/y]"
#df["aridity_class"] = "energy-limited"
#df.loc[df["Aridity"] > 1, "aridity_class"] = "water-limited"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
#g.map_dataframe(plotting_fcts.plot_coloured_scatter_random, x_name, y_name,
#                domains="aridity_class", alpha=0.1, s=10)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
g.set(xlim=[0.1, 10], ylim=[0, 2000])
#g.map(plotting_fcts.plot_origin_line, var, wtd)
#g.map(sns.rugplot, var, wtd, lw=1, alpha=.002, color="lightgrey")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

# scatter plot
x_name = "Precipitation"
y_name = "Recharge"
z_name = "Elevation"
x_unit = " [-]"
y_unit = " [-]"
#sns.set_style("whitegrid",{"grid.color": ".75", "grid.linestyle": ":"})
sns.set_style("ticks", {'axes.grid': True, "grid.color": ".85", "grid.linestyle": "-", "xtick.direction": "in", "ytick.direction": "in"})
g = sns.FacetGrid(df, col="dummy", palette='viridis', hue=z_name, height=3, aspect=2)
g.map_dataframe(sns.scatterplot, x_name, y_name, alpha=1, s=10)
#g.set(xlim=[-0.1, 0.5], ylim=[-100, 2000])
#g.map(plotting_fcts.plot_origin_line, x_name, y_name)
g.set(xlabel=x_name+x_unit, ylabel=y_name+y_unit)
g.set_titles(col_template = '{col_name}')
g.set(xscale='linear', yscale='linear')
norm = plt.Normalize(0, 2000)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
# Remove the legend and add a colorbar
g.figure.colorbar(sm)
#sns.despine(fig=g, top=False, right=False, left=False, bottom=False)
g.savefig(results_path + x_name + '_' + y_name + '_' + z_name + "_scatterplot_Moeck_colored.png", dpi=600, bbox_inches='tight')
plt.close()
"""
