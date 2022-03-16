import os
import matplotlib.pyplot as plt
import rioxarray as rxr
import time

start = time.time()
print("hello")

# specify paths
data_path = r"/home/hydrosys/data/MERIT_DEM/" #r"D:/Data/MERIT_DEM/"
results_path = "results/"

#dem_package = data_path + "dem_tif_n30e090/"
# dem_tiles = []

name = "MERIT_DEM"

# figure
fig = plt.figure(figsize=(12, 8))
ax = plt.axes() #projection=ccrs.Robinson()

# iterate over files in that directory
f_list = []
#dem_merged = []
sp_list = []

i = 0

for foldername in os.listdir(data_path):

    i = i + 1
    print(foldername + " (" + str(i) + "/57)")

    for filename in os.listdir(os.path.join(data_path, foldername)):

        dem_tmp = None
        f = os.path.join(data_path, foldername, filename)
        f_list.append(f)
        # checking if it is a file
        #if os.path.isfile(f):
        #    print(f)

        # preprocess shapefiles
        dem_tmp = rxr.open_rasterio(f, masked=True).squeeze()
        #dem_merged.append(dem_tmp)

        # plot tile
        sp_tmp = dem_tmp.plot.imshow(ax=ax, cmap='terrain', add_colorbar=False)
        sp_list.append(sp_tmp)

#dem_merged = merge_arrays(dem_merged)

ax.set(title=None) #"DEM [m]"
#ax.set_axis_off()
ax.axis('equal')
#ax.set_xlim([1, 1])
#ax.set_ylim([1, 1])
ax.set_xlabel('Lon [deg]')
ax.set_ylabel('Lat [deg]')

for sp_tmp in sp_list:
    sp_tmp.set_clim([0, 8000])

fig.colorbar(sp_tmp, ax=ax)
sp_tmp.colorbar.set_label('DEM [m]')

#plt.show()
plt.savefig(results_path + "example2_" + name + ".png", dpi=600, bbox_inches='tight')

end = time.time()
print(end - start)
