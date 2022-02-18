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

# specify paths
data_path = r"C:/Users/Sebastian/Documents/Data/" #r"C:/Users/gnann/Documents/QGIS/Topography/"
results_path = "results/" #r"C:/Users/gnann/Documents/PYTHON/Topography/results/"

shp_path = data_path + "GMBA mountain inventory V1.2(entire world)/GMBA Mountain Inventory_v1.2-World.shp"
dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
clim_path = data_path + "wc2.1_30s_bio/wc2.1_30s_bio_12.tif"

#baseline = data_path + "lines/testline.shp"
name = "NorthernAlps" #"Kilimanjaro" #"Cascades"

# create geometries
#xy_line = [37.0, 37.8, -0.3, 0.1] # Kilimanjaro
#xy_box = [35.0, 40.0, -1.0, 1.0]
#xy_line = [-125.5, -120.5, 45.0, 45.001] # Cascades
#xy_box = [-126.5, -119.5, 43.0, 47.0]
xy_line = [11.0, 11.001, 47.7, 46.7] # Northern Alps
xy_box = [10.0, 12.0, 46.0, 48.0]

# line needs to be shorter than rectangle
line = geometry.LineString([geometry.Point(xy_line[0], xy_line[2]),
                            geometry.Point(xy_line[1], xy_line[3])])
schema = {'geometry': 'LineString', 'properties': {'id': 'int'}}

# write a new shapefile
with fiona.open(results_path + 'tmp/tmp_line.shp', 'w', 'ESRI Shapefile', schema) as c:
    c.write({'geometry': mapping(line), 'properties': {'id': 123}})

baseline = results_path + 'tmp/tmp_line.shp'
line_shape = pyosp.read_shape(baseline)
lx, ly = line_shape.xy

polygon = [{'type': 'Polygon',
            'coordinates': [[[xy_box[0], xy_box[2]],
                             [xy_box[0], xy_box[3]],
                             [xy_box[1], xy_box[3]],
                             [xy_box[1], xy_box[2]],
                             [xy_box[0], xy_box[2]]]]}]

# preprocess shapefiles
#dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
#dem_clipped = dem.rio.clip(polygon, dem.rio.crs)
#dem_clipped.__array__()[np.isnan(dem_clipped.__array__())] = 1e-30
#dem_clipped.rio.to_raster(results_path + "tmp/tmp_shape.tif")
raster_dem = results_path + "tmp/tmp_shape.tif"
dem_clipped = rxr.open_rasterio(raster_dem, masked=True).squeeze()

#clim = rxr.open_rasterio(clim_path, masked=True).squeeze()
#clim_clipped = clim.rio.clip(polygon, clim.rio.crs)
#clim_clipped.__array__()[np.isnan(clim_clipped.__array__())] = 1e-30
#clim_clipped.rio.to_raster(results_path + "tmp/tmp_clim_shape.tif")
raster_clim = results_path + "tmp/tmp_clim_shape.tif"
clim_clipped = rxr.open_rasterio(raster_clim, masked=True).squeeze()

# generate swath objects
orig_dem = pyosp.Orig_curv(baseline, raster_dem, width=0.1, line_stepsize=.01, cross_stepsize=0.01)
orig_clim = pyosp.Orig_curv(baseline, raster_clim, width=0.1, line_stepsize=.01, cross_stepsize=0.01)

# plot the swath profile lines
fig = plt.figure(figsize=(6, 12), constrained_layout=True)
gs = plt.GridSpec(3, 1, figure=fig)
axes0 = fig.add_subplot(gs[0, 0])
axes1 = fig.add_subplot(gs[1, 0])
axes2 = fig.add_subplot(gs[2, 0])

swath_polylines = orig_dem.out_polylines()
#for line in swath_polylines:
#    x, y = line.xy
#    axes0.plot(x, y, color='C2')

swath_polygon = orig_dem.out_polygon()
px, py = swath_polygon.exterior.xy
axes0.plot(px, py, c='tab:orange')

#axes0.plot(lx, ly, color='C3', label="Baseline")
axes0.set_aspect('equal', adjustable='box')
#axes0.set_title("Swath profile lines")
#axes0.legend()

sp0 = dem_clipped.plot.imshow(ax=axes0, cmap='gist_earth')
axes0.set(title=None) #"DEM [m]"
#axes1.set_axis_off()
axes0.axis('equal')
axes0.set_xlim([xy_box[0], xy_box[1]])
axes0.set_ylim([xy_box[2], xy_box[3]])
axes0.set_xlabel('Lon [deg]')
axes0.set_ylabel('Lat [deg]')
sp0.colorbar.set_label('DEM [m]')
sp0.set_clim([0, 2500])

# plot swath
#orig_dem.profile_plot(ax=axes1, color='grey', label='Elevation')
#orig_clim.profile_plot(ax=axes1, color='navy', label='Precipitation')
#for i in range(len(orig_dem.dat[0])):
axes1.fill_between(orig_dem.distance, np.zeros(len(orig_dem.distance)), np.array(orig_dem.dat).mean(axis=1),
                   facecolor='tab:gray', alpha=0.8, label='Elevation')
#axes1.plot(orig_dem.distance, np.array(orig_dem.dat).mean(axis=1), c='tab:grey', label='Elevation') #np.array(orig_dem.dat)[:,i]
axes1.plot(orig_clim.distance, np.array(orig_clim.dat).mean(axis=1), c='tab:blue', label='Precipitation') #np.array(orig_dem.dat)[:,i]
# to do: add uncertainty
#axes1.legend().set_visible(False)
axes1.legend(loc='upper left')
axes1.set_xlabel('Distance [deg]')
axes1.set_ylabel('[m] / [mm/y]')
#axes1.set_ylim(0,5000)

axes2.plot(np.array(orig_clim.dat).mean(axis=1), np.array(orig_dem.dat).mean(axis=1), c='gray', alpha=0.5)
sp2 = axes2.scatter(np.array(orig_clim.dat).mean(axis=1), np.array(orig_dem.dat).mean(axis=1),
             marker='o', c=orig_dem.distance)
axes2.set_xlabel('Precipitation [mm/y]')
axes2.set_ylabel('Elevation [m]')
#axes2.set(title="Distance [deg]")
cbar2 = plt.colorbar(sp2, ax=axes2)
cbar2.ax.set_ylabel('Distance [deg]')

#fig.tight_layout()

#plt.show()
plt.savefig(results_path + "swath_" + name + ".png", dpi=600, bbox_inches='tight')