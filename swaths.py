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
name = "Cascades"

# create geometries
#xy_line = [37.0, 37.8, -0.3, 0.1] # Kilimanjaro
#xy_box = [35.0, 40.0, -1.0, 1.0]
xy_line = [-123.5, -120.5, 45.0, 45.001] # Cascades
xy_box = [-124.5, -119.5, 43.0, 47.0]

# line needs to be shorter than rectangle
line = geometry.LineString([geometry.Point(xy_line[0], xy_line[2]),
                            geometry.Point(xy_line[1], xy_line[3])])
schema = {'geometry': 'LineString', 'properties': {'id': 'int'}}

# write a new shapefile
with fiona.open(results_path + 'tmp/tmp_line.shp', 'w', 'ESRI Shapefile', schema) as c:
    ## If there are multiple geometries, put the "for" loop here
    c.write({
        'geometry': mapping(line),
        'properties': {'id': 123},
    })

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
#dem_clipped.rio.to_raster(results_path + "tmp/tmp_shape.tif")
raster_dem = results_path + "tmp/tmp_shape.tif"
dem_clipped = rxr.open_rasterio(raster_dem, masked=True).squeeze()

#clim = rxr.open_rasterio(clim_path, masked=True).squeeze()
#clim_clipped = clim.rio.clip(polygon, clim.rio.crs)
#clim_clipped.rio.to_raster(results_path + "tmp/tmp_clim_shape.tif")
raster_clim = results_path + "tmp/tmp_clim_shape.tif"
clim_clipped = rxr.open_rasterio(raster_clim, masked=True).squeeze()

# generate swath objects
orig_dem = pyosp.Orig_curv(baseline, raster_dem, width=0.5, line_stepsize=0.1, cross_stepsize=0.01)
orig_clim = pyosp.Orig_curv(baseline, raster_clim, width=0.5, line_stepsize=0.1, cross_stepsize=0.01)

# plot the swath profile lines
fig, axes = plt.subplots(2, 1, figsize=(4, 4))
swath_polylines = orig_dem.out_polylines()
#for line in swath_polylines:
#    x, y = line.xy
#    axes[0].plot(x, y, color='C1')

swath_polygon = orig_dem.out_polygon()
px, py = swath_polygon.exterior.xy
axes[0].plot(px, py, c='tab:orange')

#axes[0].plot(lx, ly, color='C3', label="Baseline")
axes[0].set_aspect('equal', adjustable='box')
#axes[0].set_title("Swath profile lines")
#axes[0].legend()

sp = dem_clipped.plot.imshow(ax=axes[0], cmap='gist_earth')
axes[0].set(title="DEM [m]")
#axes[1].set_axis_off()
axes[0].axis('equal')
axes[0].set_xlim([xy_box[0], xy_box[1]])
axes[0].set_ylim([xy_box[2], xy_box[3]])
axes[0].set_xlabel('Lon [deg]')
axes[0].set_ylabel('Lat [deg]')
#cbar = plt.colorbar(sp, ax=axes[1])
sp.set_clim([0, 2500])

# plot swath
orig_dem.profile_plot(ax=axes[1], color='grey', label='Elevation')
orig_clim.profile_plot(ax=axes[1], color='navy', label='Precipitation')
axes[1].legend().set_visible(False)
axes[1].set_xlabel('Distance [deg]')
axes[1].set_ylabel('Elev. [m] / Precip. [mm/y]')
#axes[1].set_ylim(0,5000)

fig.tight_layout()

#plt.show()
plt.savefig(results_path + "swath" + name + ".png", dpi=600, bbox_inches='tight')