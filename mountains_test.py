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

# specify paths
data_path = r"C:/Users/Sebastian/Documents/Data/" #r"C:/Users/gnann/Documents/QGIS/Topography/"
results_path = "results/" #r"C:/Users/gnann/Documents/PYTHON/Topography/results/"

dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
landform_path = data_path + r"Landforms/USGSEsriTNCWorldTerrestrialEcosystems2020/commondata/raster_data/WorldEcosystem.tif"

# open raster and plot
dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
landforms = rxr.open_rasterio(landform_path, masked=True).squeeze()

# reclassify
# create key

f, ax = plt.subplots(figsize=(10, 5))
sp = landforms.plot.imshow()
ax.set(title="Landforms")
ax.set_axis_off()
#plt.show()
sp.set_clim([0, 431])

plt.savefig(results_path + "test" ".png", dpi=600, bbox_inches='tight')

#print(landforms)