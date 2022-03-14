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
import cartopy.crs as ccrs
from rioxarray.merge import merge_arrays

# specify paths
data_path = r"D:/Data/MERIT_DEM/"
results_path = "results/"

dem_package = data_path + "dem_tif_n30e090/"
# dem_tiles = []

name = "MERIT_DEM"

# iterate over files in that directory
f_list = []
dem_merged = []
for filename in os.listdir(dem_package):
    f = os.path.join(dem_package, filename)
    f_list.append(f)
    # checking if it is a file
    #if os.path.isfile(f):
    #    print(f)

    # preprocess shapefiles
    dem_tmp = rxr.open_rasterio(f, masked=True).squeeze()
    dem_merged.append(dem_tmp)

dem_merged = merge_arrays(dem_merged)

# figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes() #projection=ccrs.Robinson()

sp0 = dem_merged.plot.imshow(ax=ax, cmap='terrain')
ax.set(title=None) #"DEM [m]"
#axes1.set_axis_off()
ax.axis('equal')
#ax.set_xlim([1, 1])
#ax.set_ylim([1, 1])
ax.set_xlabel('Lon [deg]')
ax.set_ylabel('Lat [deg]')
sp0.colorbar.set_label('DEM [m]')
#sp0.set_clim([0, np.round(np.array(orig_dem.dat).max(), 100)])
sp0.set_clim([0, 1000])

#plt.show()
plt.savefig(results_path + "example_" + name + ".png", dpi=600, bbox_inches='tight')
