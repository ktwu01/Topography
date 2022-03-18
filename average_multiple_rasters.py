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
import pandas as pd
from shapely import geometry
import pyosp
import fiona
import numpy.ma as ma

# specify paths
data_path = r"C:/Users/Sebastian/Documents/Data/" #r"C:/Users/gnann/Documents/Data/"#r"C:/Users/Sebastian/Documents/Data/" #
results_path = "results/" #r"C:/Users/gnann/Documents/PYTHON/Topography/results/"

raster_path = data_path + "/wc2.1_30s_vapr/wc2.1_30s_vapr_"
file_ending = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# loop over all rasters
raster = rxr.open_rasterio(raster_path + file_ending[0] + ".tif", masked=True).squeeze()
for i in range(1,12):
    tmp = rxr.open_rasterio(raster_path + file_ending[i] + ".tif", masked=True).squeeze()
    raster.values = raster.values + tmp.values
raster.values = raster.values/12

# save raster
raster.rio.to_raster(results_path + "wc2.1_30s_vapr_avg.tif")
