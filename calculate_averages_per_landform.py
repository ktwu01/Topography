import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
import seaborn as sns
#from functions import plotting_fcts
#from functions import get_nearest_neighbour
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
from datetime import datetime as dt
import rioxarray as rxr

data_path = "/home/hydrosys/data/resampling/" #r"D:/Data/" #

var_list = ["Landform", "pr_WorldClim", "pr_CHELSA"]
path_list = [data_path + "WorldLandform_30sec.tif",
             data_path + "P_WorldClim_30sec.tif",
             data_path + "P_CHELSA_30sec.tif"]

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

df.loc[df["pr_WorldClim"] < 0, "pr_WorldClim"] = np.nan
df.loc[df["pr_CHELSA"] > 50000, "pr_CHELSA"] = np.nan
df["pr_CHELSA"] = df["pr_CHELSA"] * 0.1
#df.loc[df["DEM"] < 0, "DEM"] = np.nan
#df["DEM"] = np.tan(np.deg2rad(df["DEM"] * 0.01))
df.loc[df["Landform"] < 1, "Landform"] = np.nan
df.loc[df["Landform"] > 4, "Landform"] = np.nan

df = df.dropna().reset_index()

# NOTE: important to remove Antarctica etc., e.g. by using one layer (slope) without Antarctica

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

print("WorldClim")
print((df["pr_WorldClim"]*df["area"]).sum()/df["area"].sum())

# merge mountains, hills, and plateaus
df.loc[df["landform"]==1, "landform"] = 5 # mountains
df.loc[df["landform"]==2, "landform"] = 5 # hills
df.loc[df["landform"]==3, "landform"] = 5 # plateaus
df.loc[df["landform"]==4, "landform"] = 6 # plains

for i in [5, 6]:
    df_tmp = df.loc[df["Landform"]==i]
    print((df_tmp["pr_WorldClim"]*df_tmp["area"]).sum()/df_tmp["area"].sum())

print("CHELSA")
print((df["pr_CHELSA"]*df["area"]).sum()/df["area"].sum())
for i in [5, 6]:
    df_tmp = df.loc[df["Landform"]==i]
    print((df_tmp["pr_CHELSA"]*df_tmp["area"]).sum()/df_tmp["area"].sum())

