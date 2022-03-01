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

shp_path = data_path + "GMBA mountain inventory V1.2(entire world)/GMBA Mountain Inventory_v1.2-World.shp"
dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
clim_path = data_path + "wc2.1_30s_bio/wc2.1_30s_bio_12.tif"

name_list = ["Cascades"]#["Sierra Nevada", "Alps", "Ecuador Andes", "France", "Himalaya", "NorthernAlps", "Kilimanjaro", "Cascades"]

for name in name_list:

    # create geometries
    import get_region
    xy_line, xy_box = get_region.get_region(name)

    line = geometry.LineString([geometry.Point(xy_line[0], xy_line[2]),
                                geometry.Point(xy_line[1], xy_line[3])])

    schema = {'geometry': 'LineString', 'properties': {'id': 'int'}}
    # write a new shapefile
    with fiona.open(results_path + 'tmp/tmp_' + name + '_line.shp', 'w', 'ESRI Shapefile', schema) as c:
        c.write({'geometry': mapping(line), 'properties': {'id': 123}})

    baseline = results_path + 'tmp/tmp_' + name + '_line.shp'
    line_shape = pyosp.read_shape(baseline)
    lx, ly = line_shape.xy

    polygon = [{'type': 'Polygon',
                'coordinates': [[[xy_box[0], xy_box[2]],
                                 [xy_box[0], xy_box[3]],
                                 [xy_box[1], xy_box[3]],
                                 [xy_box[1], xy_box[2]],
                                 [xy_box[0], xy_box[2]]]]}]

    # preprocess shapefiles
    dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
    dem_clipped = dem.rio.clip(polygon, dem.rio.crs)
    dem_clipped.__array__()[np.isnan(dem_clipped.__array__())] = -999
    dem_clipped.rio.to_raster(results_path + 'tmp/tmp_' + name + '_dem.tif')
    raster_dem = results_path + 'tmp/tmp_' + name + '_dem.tif'
    dem_clipped = rxr.open_rasterio(raster_dem).squeeze()

    clim = rxr.open_rasterio(clim_path, masked=True).squeeze()
    clim_clipped = clim.rio.clip(polygon, clim.rio.crs)
    clim_clipped.__array__()[np.isnan(clim_clipped.__array__())] = -999
    clim_clipped.rio.to_raster(results_path + 'tmp/tmp_' + name + '_clim.tif')
    raster_clim = results_path + 'tmp/tmp_' + name + '_clim.tif'
    clim_clipped = rxr.open_rasterio(raster_clim).squeeze()

    # generate swath objects
    orig_dem = pyosp.Orig_curv(baseline, raster_dem, width=0.1, line_stepsize=.01, cross_stepsize=0.01)
    orig_clim = pyosp.Orig_curv(baseline, raster_clim, width=0.1, line_stepsize=.01, cross_stepsize=0.01)

    # plot the swath profile lines
    fig = plt.figure(figsize=(12, 3), constrained_layout=True)
    gs = plt.GridSpec(1, 3, figure=fig)
    axes0 = fig.add_subplot(gs[0, 0])
    axes1 = fig.add_subplot(gs[0, 1])
    axes2 = fig.add_subplot(gs[0, 2])

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

    sp0 = dem_clipped.plot.imshow(ax=axes0, cmap='terrain')
    axes0.set(title=None) #"DEM [m]"
    #axes1.set_axis_off()
    axes0.axis('equal')
    axes0.set_xlim([xy_box[0], xy_box[1]])
    axes0.set_ylim([xy_box[2], xy_box[3]])
    axes0.set_xlabel('Lon [deg]')
    axes0.set_ylabel('Lat [deg]')
    sp0.colorbar.set_label('DEM [m]')
    #sp0.set_clim([0, np.round(np.array(orig_dem.dat).max(), 100)])
    sp0.set_clim([0, 5000])

    # plot swath
    #orig_dem.profile_plot(ax=axes1, color='grey', label='Elevation')
    #orig_clim.profile_plot(ax=axes1, color='navy', label='Precipitation')
    #for i in range(len(orig_dem.dat[0])):
    dist = orig_dem.distance
    dem_swath = np.array(orig_dem.dat)
    dem_swath[dem_swath==-999] = np.nan
    #ma.masked_invalid(dem_swath)
    clim_swath = np.array(orig_clim.dat)
    clim_swath[clim_swath==-999] = np.nan
    #ma.masked_invalid(clim_swath)

    axes1.fill_between(dist, np.zeros(len(dist)), dem_swath.mean(axis=1),
                       facecolor='tab:gray', alpha=0.25, label='Elevation')
    axes1.fill_between(dist, np.zeros(len(dist)), dem_swath.mean(axis=1)-dem_swath.std(axis=1),
                       facecolor='tab:gray', alpha=0.25)
    axes1.fill_between(dist, np.zeros(len(dist)), dem_swath.mean(axis=1)+dem_swath.std(axis=1),
                       facecolor='tab:gray', alpha=0.25)
    #axes1.plot(dist, dem_swath.mean(axis=1), c='tab:grey', label='Elevation') #np.array(orig_dem.dat)[:,i]

    axes1b = axes1.twinx()
    axes1b.plot(dist, clim_swath.mean(axis=1),
               c='tab:blue', label='Precipitation') #np.array(orig_dem.dat)[:,i]
    axes1b.fill_between(dist, clim_swath.mean(axis=1)-clim_swath.std(axis=1), clim_swath.mean(axis=1)+clim_swath.std(axis=1),
                        facecolor='tab:blue', alpha=0.25)

    lines, labels = axes1.get_legend_handles_labels()
    lines2, labels2 = axes1b.get_legend_handles_labels()
    axes1b.legend(lines + lines2, labels + labels2)
    #axes1.legend().set_visible(False)
    #axes1.legend(loc='upper left')
    axes1.set_xlabel('Distance [deg]')
    axes1.set_ylabel('Elevation [m]')
    axes1b.set_ylabel('Precipitation [mm/y]')
    axes1.set_ylim(0,5000)
    axes1b.set_ylim(0,5000)

    axes2.plot(clim_swath.mean(axis=1), dem_swath.mean(axis=1), c='grey', alpha=0.5)
    sp2 = axes2.scatter(clim_swath.mean(axis=1), dem_swath.mean(axis=1),
                 marker='o', c=dist)
    axes2.set_xlabel('Precipitation [mm/y]')
    axes2.set_ylabel('Elevation [m]')
    axes2.set_xlim([0,5000])
    axes2.set_ylim([0,5000])
    #axes2.set(title="Distance [deg]")
    cbar2 = plt.colorbar(sp2, ax=axes2)
    cbar2.ax.set_ylabel('Distance [deg]')

    #fig.tight_layout()

    #plt.show()
    plt.savefig(results_path + "swath_" + name + ".png", dpi=600, bbox_inches='tight')
