import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
from scipy import stats
import rasterio

# specify paths
data_path =  r"/home/hydrosys/data/" #r"D:/Data/" #r"C:/Users/Sebastian/Documents/Data/"
results_path = "results/" #r"C:/Users/gnann/Documents/PYTHON/Topography/results/"

#dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
landform_path = data_path + "Landforms/WorldSubLandform.tif"

# open raster and plot
#dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
landforms = rxr.open_rasterio(landform_path, masked=True).squeeze()
#landforms = rasterio.open(landform_path)
# drop na?

#h = xhistogram(da, bins=[bins])
#display(h)
#h.plot()

# histogram
f, ax = plt.subplots(figsize=(5, 5))
bins = np.linspace(.5,17.5,18)
sp = landforms.plot.hist(ax=ax, bins=bins)
ax.set(title="Histogram", ylabel="Count")
#plt.show()
for i in range(0,16):
    if i in [0,1,2,3]: # 3 is missing
        ax.patches[i].set_color('palegoldenrod')
    if i in [4,5]:
        ax.patches[i].set_color('yellowgreen')
    if i in [6,7]:
        ax.patches[i].set_color('olivedrab')
    if i in [8,9]:
        ax.patches[i].set_color('darkgoldenrod')
    if i in [10,11]:
        ax.patches[i].set_color('gray')
    if i in [12,13,14,15]:
        ax.patches[i].set_color('plum')
    if i in [16]:
        ax.patches[i].set_color('steelblue')
#ax.patches[].set_color('tab:yellow')
#ax.set_yscale('log')
ax.set_xticks([2.5, 6.5, 10.5, 14.5, 17], ["Plains", "Hills", "Mountains", "Plateaus", "Water"])
plt.savefig(results_path + "landform_histogram.png", dpi=600, bbox_inches='tight')
print("done with histogram")
plt.close()

# reclassify
bins = [0.5,4.5,6.5,8.5,10.5,12.5,16.5,17.5]

# create chunked dask version of data
data_chunked = landforms.chunk({'x': 1})
# use dask's version of digitize
import dask.array as da
reclassified = xr.apply_ufunc(da.digitize, data_chunked, bins, dask='allowed')
#reclassified = xr.apply_ufunc(np.digitize, landforms, bins)
del landforms
#reclassified = reclassified.where(reclassified != 8)
# to do: save and reload reclassified map
print("finished with reclassification")

# plot
f, ax = plt.subplots(figsize=(10, 5))
colors = ['palegoldenrod', 'yellowgreen', 'olivedrab', 'darkgoldenrod', 'gray', 'plum', 'steelblue', 'white']
class_bins = [.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(class_bins, len(colors))
sp = reclassified.plot.imshow(ax=ax, cmap=cmap, norm=norm)
sp.colorbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
ax.set(title="Landforms")
ax.set_axis_off()
ax.axis('equal')
#plt.show()
plt.savefig(results_path + "landforms.png", dpi=600, bbox_inches='tight')
plt.close()
print("done with map")
