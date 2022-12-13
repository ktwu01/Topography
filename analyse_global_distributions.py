import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from functions import plotting_fcts
from rasterio.crs import CRS
import rioxarray as rxr
import os
import seaborn as sns

# specify paths
data_path = "./data/"
results_path = "./results/"

#if not os.path.isdir(results_path):
#    os.makedirs(results_path)

def make_dataframe(data_path):
    # run on server

    source_data_path = "/home/hydrosys/data/"

    # specify file paths
    pr_path = source_data_path + "resampling/" + "P_CHELSA_30s.tif"
    pet_path = source_data_path + "resampling/" + "PET_CHELSA_30s.tif"
    slope_path = source_data_path + "resampling/" + "Slope_MERIT_30s.tif"
    elevation_path = source_data_path + "resampling/" + "Elevation_MERIT_30s.tif"
    landform_path = source_data_path + "resampling/" + "WorldLandform_30sec.tif"

    # open all datasets
    var_list = ["pr_30s", "pet_30s", "slope_30s", "elevation_30s", "landform"]
    path_list = [pr_path, pet_path, slope_path, elevation_path, landform_path]

    df = pd.DataFrame(columns=["y", "x"])
    for var, path in zip(var_list, path_list):
        rds = rxr.open_rasterio(path)
        rds = rds.squeeze().drop("spatial_ref").drop("band")
        rds.name = var
        df_tmp = rds.to_dataframe().reset_index()
        # df_tmp["y"] = np.round(df_tmp["y"],4) # because of small coordinate differences...
        # df_tmp["x"] = np.round(df_tmp["x"],4)
        df = pd.merge(df, df_tmp, on=['y', 'x'], how='outer')

    df.rename(columns={'x': 'lon', 'y': 'lat'}, inplace=True)

    # remove nodata values
    df.loc[df["pr_30s"] > 50000, "pr_30s"] = np.nan
    df.loc[df["pet_30s"] > 50000, "pet_30s"] = np.nan
    df.loc[df["slope_30s"] < 0, "slope_30s"] = np.nan
    df.loc[df["elevation_30s"] < -1000, "elevation_30s"] = np.nan
    df.loc[df["landform"] < 1, "landform"] = np.nan
    df.loc[df["landform"] > 4, "landform"] = np.nan

    # df = df.dropna()
    df = df.dropna().reset_index()

    # transform values
    df['pr_30s'] = df['pr_30s'] * 0.1
    df['pet_30s'] = df['pet_30s'] * 0.01 * 12
    df['slope_30s'] = np.tan(np.deg2rad(df['slope_30s'] * 0.01))
    df['aridity_30s'] = df['pet_30s'] / df['pr_30s']

    df.to_csv(data_path + 'global_data.csv', index=False)

    print("Done")

#make_dataframe(data_path)

df = pd.read_csv(data_path + 'global_data.csv')

df["dummy"] = ""

# categories
df.loc[df["landform"]==1, "landform"] = 5 # mountains
df.loc[df["landform"]==2, "landform"] = 5 # hills
df.loc[df["landform"]==3, "landform"] = 5 # plateaus
df.loc[df["landform"]==4, "landform"] = 6 # plains

# TODO: Account for grid cell size
df_new = []
# loop over lat,lon
for x, y in zip(df["lon"], df["lat"]):
    # https://gis.stackexchange.com/questions/421231/how-can-i-calculate-the-area-of-a-5-arcminute-grid-cell-in-square-kilometers-gi
    # 1 degree of latitude = 111.567km. This varies very slightly by latitude, but we'll ignore that
    # 5 arcminutes of latitude is 1/12 of that, so 9.297km
    # 5 arcminutes of longitude is similar, but multiplied by cos(latitude) if latitude is in radians, or cos(latitude/360 * 2 * 3.14159) if in degrees
    # we have half a degree here
    y_len = 111.567/60/2 #/12 #
    x_len = y_len * np.cos(y/360 * 2 * np.pi)
    df_new.append([x_len, y_len])
df_new = pd.DataFrame(df_new)
df_new["area"] = df_new[0]*df_new[1]
df["area"] = df_new["area"]

print("Total land area: ", str(df["area"].sum()))

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


# count above 1000m (or mountains and hills) and humid
threshold = 0.08
df_tmp = df["slope_30s"]
print("Distribution topography")
print("Arid and fraction below " + str(threshold) + ": " + str(
    round(df.loc[np.logical_and(df_tmp <= threshold,df["aridity_30s"]>1), "area"].sum() / df["area"].sum(), 2)))
print("Humid and fraction below " + str(threshold) + ": " + str(
    round(df.loc[np.logical_and(df_tmp <= threshold, df["aridity_30s"]<1), "area"].sum() / df["area"].sum(), 2)))
print("Arid and fraction above " + str(threshold) + ": " + str(
    round(df.loc[np.logical_and(df_tmp > threshold, df["aridity_30s"]>1), "area"].sum() / df["area"].sum(), 2)))
print("Humid and fraction above " + str(threshold) + ": " + str(
    round(df.loc[np.logical_and(df_tmp > threshold,df["aridity_30s"]<1), "area"].sum() / df["area"].sum(), 2)))

df_tmp = df["landform"]
print("Distribution landforms")
print("Humid and plains " + ": " + str(
    round(df.loc[np.logical_and(df["landform"]==6, df["aridity_30s"]<1), "area"].sum() / df["area"].sum(), 2)))
print("Arid and plains " + ": " + str(
    round(df.loc[np.logical_and(df["landform"]==6, df["aridity_30s"]>1), "area"].sum() / df["area"].sum(), 2)))
print("Humid and mountains " + ": " + str(
    round(df.loc[np.logical_and(df["landform"]==5, df["aridity_30s"]<1), "area"].sum() / df["area"].sum(), 2)))
print("Arid and mountains " + ": " + str(
    round(df.loc[np.logical_and(df["landform"]==5, df["aridity_30s"]>1), "area"].sum() / df["area"].sum(), 2)))


x_name = "aridity"
y_name = "slope"
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
