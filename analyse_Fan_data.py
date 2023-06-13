import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from functions import plotting_fcts
import os
import seaborn as sns

# specify paths
data_path = "./data/"
results_path = "./results/Fan/"

# Plots Fan et al. (2013) water table depth against topographic slope and calculates the fraction of observations in each landform.

if not os.path.isdir(results_path):
    os.makedirs(results_path)

def create_dataframe(data_path):
    # run on server

    data_path = "data/"
    source_data_path = "/home/hydrosys/data/"

    pr_30s_path = source_data_path + "resampling/" + "P_CHELSA_30s.tif"
    pet_30s_path = source_data_path + "resampling/" + "PET_CHELSA_30s.tif"
    slope_30s_path = source_data_path + "resampling/" + "Slope_MERIT_30s.tif"
    elevation_30s_path = source_data_path + "resampling/" + "Elevation_MERIT_30s.tif"
    landform_path = source_data_path + "resampling/" + "WorldLandform_30sec.tif"

    # open all datasets
    df = pd.read_csv(data_path + "WTD_Fan_2013.csv", sep=',')

    pr_30s = rio.open(pr_30s_path, masked=True)
    pet_30s = rio.open(pet_30s_path, masked=True)
    slope_30s = rio.open(slope_30s_path, masked=True)
    elevation_30s = rio.open(elevation_30s_path, masked=True)
    landform = rio.open(landform_path, masked=True)

    # extract point values from gridded files / shapefile
    coord_list = [(x,y) for x,y in zip(df['lon'], df['lat'])]

    df['pr_30s'] = [x for x in pr_30s.sample(coord_list)]
    df['pet_30s'] = [x for x in pet_30s.sample(coord_list)]
    df['slope_30s'] = [x for x in slope_30s.sample(coord_list)]
    df['elevation_30s'] = [x for x in elevation_30s.sample(coord_list)]
    df['landform'] = [x for x in landform.sample(coord_list)]

    # transform to np.arrays and rescale variables
    df['pr_30s'] = np.concatenate(df['pr_30s'].to_numpy())
    df['pet_30s'] = np.concatenate(df['pet_30s'].to_numpy())
    df['slope_30s'] = np.concatenate(df['slope_30s'].to_numpy())
    df['elevation_30s'] = np.concatenate(df['elevation_30s'].to_numpy())
    df['landform'] = np.concatenate(df['landform'].to_numpy())

    # remove nodata values
    df.loc[df["wtd"] > 9000, "pr_30s"] = np.nan # 9999
    df.loc[df["pr_30s"] > 50000, "pr_30s"] = np.nan # 65535
    df.loc[df["pet_30s"] > 50000, "pet_30s"] = np.nan # 65535
    df.loc[df["slope_30s"] < 0, "slope_30s"] = np.nan # -32768
    df.loc[df["elevation_30s"] < -1000, "elevation_30s"] = np.nan # -9999
    df.loc[df["landform"] < 1, "landform"] = np.nan
    df.loc[df["landform"] > 4, "landform"] = np.nan

    # transform values
    df['pr_30s'] = df['pr_30s'] * 0.1
    df['pet_30s'] = df['pet_30s'] * 0.01 * 12
    df['slope_30s'] = np.tan(np.deg2rad(df['slope_30s'] * 0.01))

    df['aridity_30s'] = df['pet_30s']/df['pr_30s']

    # save to csv file
    df.to_csv(data_path + 'wtd_data.csv', index=False)

    print("Done")

#create_dataframe(data_path)

# plot data
df = pd.read_csv(data_path + 'wtd_data.csv')
df = df.dropna()
df["dummy"] = ""

# reclassify landforms
df.loc[df["landform"]==1, "landform"] = 5 # mountains
df.loc[df["landform"]==2, "landform"] = 5 # hills
df.loc[df["landform"]==3, "landform"] = 5 # plateaus
df.loc[df["landform"]==4, "landform"] = 6 # plains

# slope and wtd
x_name = "slope_30s"
y_name = "wtd"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name, color="silver", marker='o', lw=0, alpha=0.01, s=1, label=None)
g.set(xlim=[0.0001, 1], ylim=[0.1, 100])
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="dummy", group="")
#g.add_legend(loc=(.2, .75), handletextpad=0.0)
g.set(xlabel = "Slope [-]", ylabel = "Water Table Depth [m]")
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='log')
plt.savefig(results_path + x_name + '_' + y_name + ".png", dpi=600, bbox_inches='tight')
plt.close()

# landform distribution
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
