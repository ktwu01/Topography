import matplotlib as mpl
import matplotlib.pyplot as plt
import rioxarray as rxr
import time

# specify paths
data_path = r"/home/hydrosys/data/" # r"D:/Data/" #r"C:/Users/Sebastian/Documents/Data/"#
results_path = "../results/"  #r"C:/Users/gnann/Documents/PYTHON/Topography/results/"

#dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
landform_path = data_path + "Landforms/WorldSubLandform.tif"

# open raster and plot
#dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
landforms = rxr.open_rasterio(landform_path, chunks="auto").squeeze()#, masked=True)#.squeeze()
#landforms = rasterio.open(landform_path)
# drop na?
#data_chunked = landforms.chunk({"x": 1000, "y": 1000})
print("done with data loading")

"""
start_time = time.time()

# compute histogram
bins = np.linspace(.5,17.5,18)
h, bins = da.histogram(landforms, bins=bins)
h = h.compute()

# plot histogram
f, ax = plt.subplots(figsize=(5, 5))
#sp = landforms.plot.hist(ax=ax, bins=bins)
sp = ax.bar((bins[0:-1]+bins[1:])/2, h)
ax.set(title="Histogram", ylabel="Count")
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

# save histogram
plt.savefig(results_path + "landform_histogram.png", dpi=600, bbox_inches='tight')
plt.close()

print("done with histogram ")
end_time = time.time()
print(end_time - start_time)
"""

"""
start_time = time.time()
# reclassify
bins = [0.5,4.5,6.5,8.5,10.5,12.5,16.5,17.5]
#bins = [0,4,6,8,10,12,16,17]

# use dask's version of digitize
reclassified = xr.apply_ufunc(da.digitize, landforms, bins, dask='allowed')
#reclassified = reclassified.compute()
#del landforms
#reclassified = reclassified.astype('int8')
#todo: make file smaller... check data format
reclassified.rio.to_raster(results_path + "reclassified_landforms.tif", dtype="int8")
print("done with saving reclassification")
end_time = time.time()
print(end_time - start_time)
del landforms, reclassified
"""

# plot
start_time = time.time()
reclassified = rxr.open_rasterio(results_path + "reclassified_landforms.tif").squeeze()#, masked=True)#.squeeze()

f, ax = plt.subplots(figsize=(10, 5))
colors = ['palegoldenrod', 'yellowgreen', 'olivedrab', 'darkgoldenrod', 'gray', 'plum', 'steelblue', 'white']
class_bins = [.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(class_bins, len(colors))
sp = reclassified.plot.imshow(ax=ax, cmap=cmap, norm=norm)
print("done with imshow")
sp.colorbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
ax.set(title="Landforms")
ax.set_axis_off()
ax.axis('equal')
#plt.show()
plt.savefig(results_path + "landforms.png", dpi=600, bbox_inches='tight')
plt.close()

print("done with map")
end_time = time.time()
print(end_time - start_time)

