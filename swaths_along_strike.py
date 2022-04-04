import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
from scipy import stats
import pandas as pd
from shapely import geometry
import shapely
import pyosp
import fiona
from matplotlib.pyplot import cm

# specify paths
data_path = r"D:/Data/" #r"C:/Users/Sebastian/Documents/Data/" #r"C:/Users/gnann/Documents/Data/"#
results_path = "results/" #r"C:/Users/gnann/Documents/PYTHON/Topography/results/"

shp_path = data_path + "GMBA mountain inventory V1.2(entire world)/GMBA Mountain Inventory_v1.2-World.shp"
dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
clim_path = data_path + "wc2.1_30s_bio/wc2.1_30s_bio_12.tif"
clim2_path = data_path + "wc2.1_30s_vapr/wc2.1_30s_vapr_avg.tif"

# create smooth lines in QGIS, if possible based on objective criteria (watershed boundaries etc.)
name_list = ["European_Alps", "Ecuadorian_Andes", "Himalaya", "Cascades"]

for name in name_list:

    from get_region_data import get_region
    line_path, xlim, ylim = get_region(name)

    # check if folder exists
    if not os.path.isdir(results_path + name + "/"):
        os.makedirs(results_path + name + "/")

    # remove all files in folder
    for f in os.listdir(results_path + name + "/"):
        os.remove(os.path.join(results_path + name + "/", f))

    # create geometries

    line = pyosp.read_shape(line_path)
    #mountain_shp = gpd.read_file(shp_path)
    #mountain_range = mountain_shp.loc[mountain_shp.Name==name]

    # preprocess shapefiles
    dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
    clim = rxr.open_rasterio(clim_path, masked=True).squeeze()
    clim2 = rxr.open_rasterio(clim2_path, masked=True).squeeze()

    # swath dimensions
    d = 1.0 # length of swath
    w = 1.0 # width
    distances = np.arange(0, line.length, w)[:-1]
    # or alternatively without NumPy:
    # points_count = int(line.length // d) + 1
    # distances = (distance_delta * i for i in range(points_count))
    points = [line.interpolate(distance) for distance in distances] + [line.boundary[1]]
    #mp = shapely.ops.unary_union(points)  # or new_line = LineString(points)
    from shapely.geometry import Point, MultiPoint
    mp = MultiPoint(list(points))

    # plot the swath profile lines
    fig = plt.figure(figsize=(6, 3), constrained_layout=True)
    axes = plt.axes()

    sp0 = dem.plot.imshow(ax=axes, cmap='terrain')
    axes.set(title=None)  # "DEM [m]"
    # axes.set_axis_off()
    # axes.axis('equal')
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xlabel('Lon [deg]')
    axes.set_ylabel('Lat [deg]')
    sp0.colorbar.set_label('DEM [m]')
    # sp0.set_clim([0, np.round(np.array(orig_dem.dat).max(), 100)])
    sp0.set_clim([0, 5000])  # 100*round(np.max(dem.values/100))

    x, y = line.xy
    axes.plot(x, y, color='tab:red')

    # create plot for elevation profiles
    fig2 = plt.figure(figsize=(4, 4), constrained_layout=True)
    axes2 = plt.axes()

    color = iter(cm.plasma(np.linspace(0, 1, len(mp))))

    # loop over swaths
    for p in range(0, len(mp)-1):
        nextcolor = next(color)

        print('')
        print(p)

        xs = [point.x for point in mp]
        ys = [point.y for point in mp]
        xx = (np.array(xs[1:])+np.array(xs[0:-1]))/2
        yy = (np.array(ys[1:])+np.array(ys[0:-1]))/2

        import perp_pts
        m = (ys[p+1] - ys[p]) / (xs[p+1] - xs[p])
        x1, y1, x2, y2 = perp_pts.perp_pts(xx[p], yy[p], m, d, [xs[p], ys[p], xs[p+1], ys[p+1]])

        """
        plt.scatter(xs, ys)
        plt.plot(x, y)
        plt.scatter(xx, yy)
        plt.scatter(x1,y1)
        plt.scatter(x2,y2)
        plt.show()
        plt.axis('equal')
        """

        # write line (typically goes from north to south - curved lines can make this a bit tricky...)
        line = geometry.LineString([geometry.Point(x2, y2),
                                    geometry.Point(x1, y1)]) #xx[p], yy[p]

        schema = {'geometry': 'LineString', 'properties': {'id': 'int'}}
        # write a new shapefile
        with fiona.open(results_path + 'tmp/tmp_' + name + '_line.shp', 'w', 'ESRI Shapefile', schema) as c:
            c.write({'geometry': mapping(line), 'properties': {'id': 123}})

        baseline = results_path + 'tmp/tmp_' + name + '_line.shp'
        line_shape = pyosp.read_shape(baseline)
        lx, ly = line_shape.xy

        # generate swath objects
        line_stepsize = 0.05
        cross_stepsize = 0.05
        orig_dem = pyosp.Orig_curv(baseline, dem_path, width=w, line_stepsize=line_stepsize, cross_stepsize=cross_stepsize)
        orig_clim = pyosp.Orig_curv(baseline, clim_path, width=w, line_stepsize=line_stepsize, cross_stepsize=cross_stepsize)
        orig_clim2 = pyosp.Orig_curv(baseline, clim2_path, width=w, line_stepsize=line_stepsize, cross_stepsize=cross_stepsize)

        swath_polylines = orig_dem.out_polylines()
        #for line in swath_polylines:
        #    x, y = line.xy
        #    axes0.plot(x, y, color='C2')

        swath_polygon = orig_dem.out_polygon()
        px, py = swath_polygon.exterior.xy
        axes.plot(px, py, c=nextcolor)
        """
        if p % 10 == 0:
            axes.plot(px, py, c='tab:purple')
        else:
            axes.plot(px, py, c='tab:orange')
        """
        #axes.plot(lx, ly, color='C3', label="Baseline")
        axes.set_aspect('equal', adjustable='box')
        #axes.set_title("Swath profile lines")
        #axes.legend()

        ###
        try:
            # plot swath
            #orig_dem.profile_plot(ax=axes1, color='grey', label='Elevation')
            #orig_clim.profile_plot(ax=axes1, color='navy', label='Precipitation')
            #for i in range(len(orig_dem.dat[0])):
            dist = orig_dem.distance
            dem_swath = np.array(orig_dem.dat)
            if (len(dist) == len(dem_swath) + 1): # sometimes dist is longer than swath
                dist = orig_dem.distance[0:-1]
            dem_swath[dem_swath==-32768.] = np.nan # Note: works only because this is returned as nodata value
            isnan = np.isnan(dem_swath).any(axis=1)
            dem_swath = dem_swath[~isnan]
            #ma.masked_invalid(dem_swath)
            clim_swath = orig_clim.dat
            clim_swath = [d for (d, remove) in zip(clim_swath, isnan) if not remove]
            clim_swath = np.array(clim_swath)
            #clim_swath[clim_swath==-999] = np.nan
            #ma.masked_invalid(clim_swath)
            clim2_swath = orig_clim2.dat
            clim2_swath = [d for (d, remove) in zip(clim2_swath, isnan) if not remove]
            clim2_swath = np.array(clim2_swath)
            #clim2_swath[clim2_swath==-999] = np.nan
            clim2_swath = clim2_swath*1000 # transform to Pa
            #ma.masked_invalid(clim_swath)
            dist = dist[~isnan]

            # plot the swath profile lines
            fig1 = plt.figure(figsize=(8, 3), constrained_layout=True)
            axes1 = plt.axes()

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

            axes1b.plot(dist, clim2_swath.mean(axis=1),
                        c='tab:green', label='Vapor pressure')  # np.array(orig_dem.dat)[:,i]
            axes1b.fill_between(dist, clim2_swath.mean(axis=1) - clim2_swath.std(axis=1),
                                clim2_swath.mean(axis=1) + clim2_swath.std(axis=1),
                                facecolor='tab:green', alpha=0.25)


            lines, labels = axes1.get_legend_handles_labels()
            lines2, labels2 = axes1b.get_legend_handles_labels()
            axes1b.legend(lines + lines2, labels + labels2)
            #axes1.legend().set_visible(False)
            #axes1.legend(loc='upper left')
            axes1.set_xlabel('Distance [deg]')
            axes1.set_ylabel('Elevation [m]')
            axes1b.set_ylabel('Precipitation [mm/y] / Vapor pressure [Pa]')
            #axes1.set_ylim(0,5000)
            #axes1b.set_ylim(0,5000)

            #plt.show()
            fig1.savefig(results_path + name + "/" + "swath_along_strike_profiles_" + str(p+1) + "_" + name + ".png", dpi=600, bbox_inches='tight')
            plt.close(fig1)

        except:
            print('')
            print(p)

        # plot elevation profile
        # todo: use binning
        axes2.plot(clim_swath.mean(axis=1), dem_swath.mean(axis=1), color=nextcolor, alpha=0.75)
        """
        if p % 10 == 0:
            axes2.plot(clim_swath.mean(axis=1), dem_swath.mean(axis=1),
                       color='tab:purple', alpha=0.5)
        else:
            axes2.plot(clim_swath.mean(axis=1), dem_swath.mean(axis=1),
                       color='tab:orange', alpha=0.5)
        """

    # plt.show()
    fig.savefig(results_path + name + "/" + "swaths_along_strike_" + name + ".png", dpi=600, bbox_inches='tight')
    plt.close(fig)

    # plt.show()
    axes2.set_xlabel('Precipitation [mm/y]')
    axes2.set_ylabel('Elevation [km]')
    fig2.savefig(results_path + name + "/" + "swaths_elevation_profiles_" + name + ".png", dpi=600, bbox_inches='tight')
    plt.close(fig2)
