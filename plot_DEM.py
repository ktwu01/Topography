import os
import matplotlib.pyplot as plt
import rioxarray as rxr
import time

# todo: dask

# time script run
start = time.time()
print(start)

# specify paths
data_path = r"/home/hydrosys/data/MERIT_DEM/" #r"D:/Data/MERIT_DEM/"#
results_path = "results/"

name = "MERIT_DEM"

# figure
#fig = plt.figure()
#ax = plt.axes(projection=ccrs.Robinson())
#ax.set_global()
fig = plt.figure(figsize=(12, 6))
ax = plt.axes()

# iterate over files in that directory
f_list = []
# dem_merged = []
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
        sp_tmp = dem_tmp.plot.imshow(ax=ax, cmap='gist_earth', add_colorbar=False)
        sp_tmp.set_clim([0, 5000])
        sp_list.append(sp_tmp)

    #if i > 3: # for testing script
    #    break

ax.set(title=None) #"DEM [m]"
#ax.set_axis_off()
ax.axis('equal')
ax.set_xlim([-180, 180])
ax.set_ylim([-60, 90])
ax.set_xlabel('Lon [deg]')
ax.set_ylabel('Lat [deg]')

for sp_tmp in sp_list:
    sp_tmp.set_clim([0, 5000])

fig.colorbar(sp_tmp, ax=ax)
sp_tmp.colorbar.set_label('DEM [m]')

#box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
#x0, y0, x1, y1 = box.bounds
#ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())

#plt.show()
plt.savefig(results_path + "example2_" + name + ".png", dpi=600, bbox_inches='tight')

end = time.time()
print(end - start)
