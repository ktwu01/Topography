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
data_path = "./data/"
slope_path = "D:/Data/DEMs/Geomorpho90m/" + "dtm_slope_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
twi_path = "D:/Data/DEMs/Geomorpho90m/" + "dtm_cti_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
twi_path = "D:/Data/DEMs/Geomorpho90m/" + "dtm_cti_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
#twi_path = "C:/Users/gnann/Desktop/dtm_twi_merit.dem_m_1km_s0..0cm_2017_v1.0.tif"
elevation_path = "D:/Data/DEMs/MERIT_250m/" + "dtm_elevation_merit.dem_m_250m_s0..0cm_2017_v1.0.tif"
pr_path = "D:/Data/CHELSA/CHELSA_bio12_1981-2010_V.2.1.tif"
pet_path = "D:/Data/CHELSA/CHELSA_pet_penman_mean_1981-2010_V.2.1.tif"

slope = rio.open(slope_path, masked=True)
twi = rio.open(twi_path, masked=True)
elevation = rio.open(elevation_path, masked=True)
pr = rio.open(pr_path, masked=True)
pet = rio.open(pet_path, masked=True)

# check if folder exists
results_path = "./results/moeck/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
df = pd.read_csv(data_path + "global_groundwater_recharge_moeck-et-al.csv", sep=',')

#df_domains = pd.read_csv("./2b/aggregated/domains.csv", sep=',')
#geometry = [Point(xy) for xy in zip(df_domains.lon, df_domains.lat)]
#gdf_domains = gpd.GeoDataFrame(df_domains, geometry=geometry)

#closest = get_nearest_neighbour.nearest_neighbor(gdf, gdf_domains, return_dist=True)
#closest = get_nearest_neighbour.nearest_neighbor(gdf_domains, gdf, return_dist=True)
#closest = closest.rename(columns={'geometry': 'closest_geom'})
# merge the datasets by index (for this, it is good to use '.join()' -function)
#df = gdf.join(closest)

df.rename(columns={'Groundwater recharge [mm/y]': 'Recharge',
                   'Longitude': 'lon', 'Latitude': 'lat'}, inplace=True)

coord_list = [(x, y) for x, y in zip(df['lon'], df['lat'])]

df['Slope'] = [x for x in slope.sample(coord_list)]
df['Slope'] = np.concatenate(df['Slope'].to_numpy())
df.loc[df["Slope"] < 0, "Slope"] = np.nan
df['Slope'] = np.tan(np.deg2rad(df['Slope'] * 0.01))

df['TWI'] = [x for x in twi.sample(coord_list)]
df['TWI'] = np.concatenate(df['TWI'].to_numpy())
df.loc[df["TWI"] < -10000, "TWI"] = np.nan
df['TWI'] = df['TWI'] * 0.001

df['Elevation'] = [x for x in elevation.sample(coord_list)]
df['Elevation'] = np.concatenate(df['Elevation'].to_numpy())
df.loc[df["Elevation"] < -1000, "Elevation"] = np.nan

df['Precipitation'] = [x for x in pr.sample(coord_list)]
df['Precipitation'] = np.concatenate(df['Precipitation'].to_numpy())
df.loc[df["Precipitation"] > 50000, "Precipitation"] = np.nan
df['Precipitation'] = df['Precipitation'] * 0.1

df['Potential Evapotranspiration'] = [x for x in pet.sample(coord_list)]
df['Potential Evapotranspiration'] = np.concatenate(df['Potential Evapotranspiration'].to_numpy())
df.loc[df["Potential Evapotranspiration"] > 50000, "Potential Evapotranspiration"] = np.nan
df['Potential Evapotranspiration'] = df['Potential Evapotranspiration'] * 0.1

df["Recharge Ratio"] = df["Recharge"]/df["Precipitation"]
df["Aridity"] = df["Potential Evapotranspiration"]/df["Precipitation"]

df["dummy"] = ""

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
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
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
x_name = "TWI"
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
g.set(xlim=[-10, 10], ylim=[0, 1500])
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
