import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
#from scipy import stats
import seaborn as sns
from functions import plotting_fcts
#from functions import get_nearest_neighbour
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio
from scipy import stats

# This script ...

# prepare data
data_path = "/home/hydrosys/data/" #data_path = r"D:/Data/"
results_path = "results/"


# use 5min data
pr_path = data_path + "resampling/" + "P_CHELSA_5min.tif"
pet_path = data_path + "resampling/" + "PET_CHELSA_5min.tif"
slope_path = data_path + "resampling/" + "Slope_MERIT_5min.tif"
elevation_path = data_path + "resampling/" + "Elevation_MERIT_5min.tif"
permeability_path = data_path + "resampling/" + "Permeability_Huscroft_2018_5min.tif"
wtr_path = data_path + "resampling/" + "WTR_Cuthbert_2019_5min.tif"

pr = rio.open(pr_path, masked=True)
pet = rio.open(pet_path, masked=True)
slope = rio.open(slope_path, masked=True)
elevation = rio.open(elevation_path, masked=True)
permeability = rio.open(permeability_path, masked=True)
wtr = rio.open(wtr_path, masked=True)

# check if folder exists
results_path = "./results/markovich/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
df = pd.read_csv("data/Markovich_Table_1.csv", sep=';')
df.rename(columns={'xxx': 'xxx', 'Lon': 'lon', 'Lat': 'lat'}, inplace=True)

coord_list = [(x, y) for x, y in zip(df['lon'], df['lat'])]

df['Precipitation'] = [x for x in pr.sample(coord_list)]
df['Precipitation'] = np.concatenate(df['Precipitation'].to_numpy())
df.loc[df["Precipitation"] > 50000, "Precipitation"] = np.nan
df['Precipitation'] = df['Precipitation'] * 0.1

df['Potential Evapotranspiration'] = [x for x in pet.sample(coord_list)]
df['Potential Evapotranspiration'] = np.concatenate(df['Potential Evapotranspiration'].to_numpy())
df.loc[df["Potential Evapotranspiration"] > 50000, "Potential Evapotranspiration"] = np.nan
df['Potential Evapotranspiration'] = df['Potential Evapotranspiration'] * 0.01 * 12

df['Slope'] = [x for x in slope.sample(coord_list)]
df['Slope'] = np.concatenate(df['Slope'].to_numpy())
df.loc[df["Slope"] < 0, "Slope"] = np.nan
df['Slope'] = np.tan(np.deg2rad(df['Slope'] * 0.01))

df['Elevation'] = [x for x in elevation.sample(coord_list)]
df['Elevation'] = np.concatenate(df['Elevation'].to_numpy())
df.loc[df["Elevation"] < -1000, "Elevation"] = np.nan

df['Permeability'] = [x for x in permeability.sample(coord_list)]
df['Permeability'] = np.concatenate(df['Permeability'].to_numpy())
#df.loc[df["Permeability"] < -1000, "Permeability"] = np.nan

df['WTR'] = [x for x in wtr.sample(coord_list)]
df['WTR'] = np.concatenate(df['WTR'].to_numpy())
#df.loc[df["Permeability"] < -1000, "Permeability"] = np.nan

df["Aridity"] = df["Potential Evapotranspiration"]/df["Precipitation"]

df["dummy"] = ""

#
x_name = "Elevation"
y_name = "Recharge"
x_unit = " [m]"
y_unit = " [%]"
#df["aridity_class"] = "energy-limited"
#df.loc[df["Aridity"] > 1, "aridity_class"] = "water-limited"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw=0, color='lightgrey')
g.set(xlim=[0, 3000], ylim=[0, 100])
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='linear', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Slope"
y_name = "Recharge"
x_unit = " [-]"
y_unit = " [%]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
g.set(xlim=[0.001, 1], ylim=[0, 100])
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Precipitation"
y_name = "Recharge"
x_unit = " [mm/y]"
y_unit = " [%]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
g.set(xlim=[0, 3000], ylim=[0, 100])
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='linear', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()


x_name = "Potential Evapotranspiration"
y_name = "Recharge"
x_unit = " [mm/y]"
y_unit = " [%]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
g.set(xlim=[500, 2000], ylim=[0, 100])
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='linear', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

x_name = "Aridity"
y_name = "Recharge"
x_unit = " [-]"
y_unit = " [%]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
g.set(xlim=[0.1, 10], ylim=[0, 100])
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()


x_name = "Permeability"
y_name = "Recharge"
x_unit = " [-]"
y_unit = " [%]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
g.set(xlim=[-1700, 1000], ylim=[0, 100])
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='linear', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()


x_name = "WTR"
y_name = "Recharge"
x_unit = " [-]"
y_unit = " [%]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name,  alpha=1, s=10, lw = 0, color='lightgrey')
g.set(xlim=[-3, 3], ylim=[0, 100])
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='linear', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

df = df.drop(labels=[8,9], axis=0)
x_name = "Elevation"
y_name = "Recharge"
z_name = "Aridity"
u_name = "Permeability"
x_unit = " [-]"
y_unit = " [%]"
df["hue"] = np.round(df[z_name],2) # to have fewer unique values
df["size"] = np.round(df[y_name],0) # to have fewer unique values
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", hue="hue", palette="viridis", col_wrap=4)
#g.map_dataframe(sns.scatterplot, x_name, u_name)
plt.scatter((df[x_name]), df[u_name], c=df[z_name], s=df[y_name]*5, lw = 0)
#g.set(xlim=[0.001, 1], ylim=[-1700, -1000])
#g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
#g.set_titles(col_template='{col_name}')
#g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_color.png", dpi=600, bbox_inches='tight')
plt.close()

r_sp1, _ = stats.spearmanr(df["Recharge"], df["Elevation"], nan_policy='omit')
print(str(np.round(r_sp1,2)))
r_sp1, _ = stats.spearmanr(df["Recharge"], df["Slope"], nan_policy='omit')
print(str(np.round(r_sp1,2)))
r_sp1, _ = stats.spearmanr(df["Recharge"], df["Aridity"], nan_policy='omit')
print(str(np.round(r_sp1,2)))
r_sp1, _ = stats.spearmanr(df["Recharge"], df["Permeability"], nan_policy='omit')
print(str(np.round(r_sp1,2)))
r_sp1, _ = stats.spearmanr(df["Recharge"], df["WTR"], nan_policy='omit')
print(str(np.round(r_sp1,2)))
r_sp1, _ = stats.spearmanr(df["Elevation"], df["Permeability"], nan_policy='omit')
print(str(np.round(r_sp1,2)))

