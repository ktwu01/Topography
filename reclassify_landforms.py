import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from scipy import stats

# specify paths
data_path = r"D:/Data/" #r"C:/Users/Sebastian/Documents/Data/"
results_path = "results/" #r"C:/Users/gnann/Documents/PYTHON/Topography/results/"

dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
landform_path = data_path + "" #todo: add reclassified landforms

# open raster and plot
dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
landforms = rxr.open_rasterio(landform_path, masked=True).squeeze()

#todo: add histogram

# plot
f, ax = plt.subplots(figsize=(10, 5))
sp = landforms.plot.imshow()
ax.set(title="Landforms")
ax.set_axis_off()
#plt.show()
#sp.set_clim([0, 431])
plt.savefig(results_path + "landforms.png", dpi=600, bbox_inches='tight')

#print(landforms)

"""
import json
import subprocess

dataset_uri = data_path + r"Landforms/USGSEsriTNCWorldTerrestrialEcosystems2020/commondata/raster_data/WorldEcosystem.tif"
_rat = subprocess.check_output('gdalinfo -json ' + dataset_uri, shell=True)
data = json.loads(_rat) # load json string into dictionary
print(data)

# to get band-level data
bands = data['bands']
"""