import os
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import pyosp
from matplotlib.pyplot import cm
from functions.get_geometries import get_strike_geometries
from functions.get_perp_pts import perp_pts
from functions.create_shapefiles import create_line_shp
from functions.get_swath_data import get_swath_data
from functions.get_geometries import get_swath_indices_new
from scipy import stats
from shapely.geometry import MultiPoint
from sklearn.linear_model import LinearRegression

# Creates plots of elevation vs. the climatic water balance for several swaths in different mountain regions.

# specify paths
data_path = r"C:/Users/Sebastian/Documents/Data/"
data_path = r"D:/Data/"
results_path = "results/"

dem_path = data_path + "WorldClim/wc2.1_30s_elev/wc2.1_30s_elev.tif"
pr_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_12.tif"
pet_path = data_path + "WorldClim/7504448/global-et0_annual.tif/et0_yr/et0_yr.tif"
t_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_1.tif"
#dem_path = data_path + "DEMs/MERIT_250m/Elevation_MERIT_30s.tif"
#pr_path = data_path + "CHELSA/CHELSA_bio12_1981-2010_V.2.1.tif"
#pet_path = data_path + "CHELSA/CHELSA_pet_penman_mean_1981-2010_V.2.1.tif"
#t_path = data_path + "CHELSA/CHELSA_bio1_1981-2010_V.2.1.tif"

# create smooth lines in QGIS, if possible based on objective criteria (watershed boundaries etc.)
#name_list = ["Cordillera Central Ecuador", "Himalaya", "Sierra Madre del Sur", "Ethiopian Highlands",
#             "Southern Andes", "Sierra Nevada", "European Alps", "Pyrenees"]#
name_list = ["Cordillera Central Ecuador", "Himalaya", "Sierra Nevada", "European Alps"]

# load dem shapefile
dem = rxr.open_rasterio(dem_path, masked=True).squeeze() #todo: remove masked ...

for name in name_list:

    print(name)

    # check if folders exist
    path = results_path + name + "/shapefiles/"
    if not os.path.isdir(path):
        os.makedirs(path)

    path = results_path + name + "/swaths_elevation_profiles/"
    if not os.path.isdir(path):
        os.makedirs(path)

    # remove all files in folder
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

    line_path, xlim, ylim = get_strike_geometries(name)
    swath_ind, forcinglim = get_swath_indices_new(name)

    # create line
    line = pyosp.read_shape(line_path)

    # swath dimensions
    d = 2.0 # length of swath
    w = 0.5 # width
    distances = np.arange(0, line.length, w)[:-1]
    points = [line.interpolate(distance) for distance in distances] + [line.boundary[1]]
    mp = MultiPoint(list(points))

    ### PLOT 1 ###
    # plot the swaths on top of the DEM
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)
    axes = plt.axes()

    sp0 = dem.plot.imshow(ax=axes, cmap='gray')
    axes.set(title=None)  # "DEM [m]"
    # axes.set_axis_off()
    axes.axis('equal')
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xlabel('Lon [deg]')
    axes.set_ylabel('Lat [deg]')
    sp0.colorbar.set_label('DEM [m]')
    sp0.set_clim([0, 3000])  # 100*round(np.max(dem.values/100))

    x, y = line.xy
    #axes.plot(x, y, color='silver')

    color = iter(cm.plasma(np.linspace(0, 1, len(swath_ind)*2)))

    # loop over swaths
    for p in range(0, len(mp)-1):

        print('')
        print(p)

        xs = [point.x for point in mp]
        ys = [point.y for point in mp]
        xx = (np.array(xs[1:])+np.array(xs[0:-1]))/2
        yy = (np.array(ys[1:])+np.array(ys[0:-1]))/2

        m = (ys[p+1] - ys[p]) / (xs[p+1] - xs[p])
        x1, y1, x2, y2 = perp_pts(xx[p], yy[p], m, d, [xs[p], ys[p], xs[p+1], ys[p+1]])

        # create line (typically goes from north to south - curved lines can make this a bit tricky...)
        baseline = results_path + name + '/shapefiles/line_tmp.shp'
        #create_line_shp([x2, x1, y2, y1], baseline)
        if name in ["Cordillera Central Ecuador", "Sierra Nevada", "Himalaya"]:
            create_line_shp([x1, xx[p], y1, yy[p]], baseline)
        #elif name in ["Himalaya"]:
        #    create_line_shp([x2, x1, y2, y1], baseline)
        else:
            create_line_shp([x2, xx[p], y2, yy[p]], baseline)

        line_shape = pyosp.read_shape(baseline)
        lx, ly = line_shape.xy

        # generate swath objects
        line_stepsize = 0.05
        cross_stepsize = 0.05
        orig_dem = pyosp.Orig_curv(baseline, dem_path, width=w, line_stepsize=line_stepsize, cross_stepsize=cross_stepsize)
        orig_pr = pyosp.Orig_curv(baseline, pr_path, width=w, line_stepsize=line_stepsize, cross_stepsize=cross_stepsize)
        orig_pet = pyosp.Orig_curv(baseline, pet_path, width=w, line_stepsize=line_stepsize, cross_stepsize=cross_stepsize)
        orig_t = pyosp.Orig_curv(baseline, t_path, width=w, line_stepsize=line_stepsize, cross_stepsize=cross_stepsize)

        swath_polylines = orig_dem.out_polylines()
        #for line in swath_polylines:
        #    x, y = line.xy
        #    axes0.plot(x, y, color='C2')

        x, y = line.xy
        axes.plot(x, y, color='tab:red')

        swath_polygon = orig_dem.out_polygon()
        px, py = swath_polygon.exterior.xy

        if p in swath_ind:
            nextcolor = next(color)
            axes.plot(px, py, c='tab:orange') #nextcolor
        else:
            pass
            #axes.plot(px, py, c='silver')

        dist, dem_swath, pr_swath, pet_swath, t_swath = \
            get_swath_data(orig_dem, orig_pr, orig_pet, orig_t, line_shape)
        # account for offset and scale (only for CHELSA)
        #pr_swath = pr_swath * 0.1
        #pet_swath = pet_swath * 0.01 * 12
        #t_swath = t_swath * 0.1 - 273.15

        # plot elevation profile
        if p in swath_ind:

            ### PLOT 2 ###
            # aridity elevation profiles
            fig2 = plt.figure(figsize=(2, 2), constrained_layout=True)
            axes2 = plt.axes()

            ### PLOT 3 ###
            # PET and P elevation profiles
            fig3 = plt.figure(figsize=(2, 2), constrained_layout=True)
            axes3 = plt.axes()

            # aridity
            aridity = (pet_swath/pr_swath).flatten()
            n_bins = 10
            bin_edges = np.linspace(np.min(dem_swath.flatten()), np.max(dem_swath.flatten()), n_bins+1)
            bin_medians = (bin_edges[1:]+bin_edges[:-1])/2 # actually means...
            mean_stat = stats.binned_statistic(dem_swath.flatten(), aridity, statistic=lambda y: np.nanmean(y), bins=bin_edges)
            std_stat = stats.binned_statistic(dem_swath.flatten(), aridity, statistic=lambda y: np.nanstd(y), bins=bin_edges)

            # PET and P
            mean_stat_PET = stats.binned_statistic(dem_swath.flatten(), pet_swath.flatten(), statistic=lambda y: np.nanmean(y), bins=bin_edges)
            std_stat_PET = stats.binned_statistic(dem_swath.flatten(), pet_swath.flatten(), statistic=lambda y: np.nanstd(y), bins=bin_edges)
            mean_stat_P = stats.binned_statistic(dem_swath.flatten(), pr_swath.flatten(), statistic=lambda y: np.nanmean(y), bins=bin_edges)
            std_stat_P = stats.binned_statistic(dem_swath.flatten(), pr_swath.flatten(), statistic=lambda y: np.nanstd(y), bins=bin_edges)

            axes2.plot(mean_stat.statistic, bin_medians, color='tab:green') #nextcolor
            axes2.fill_betweenx(bin_medians, mean_stat.statistic - std_stat.statistic,
                             mean_stat.statistic + std_stat.statistic, facecolor='tab:green', alpha=0.25) #nextcolor


            axes3.plot(mean_stat_PET.statistic, bin_medians, color='tab:orange')
            axes3.fill_betweenx(bin_medians, mean_stat_PET.statistic - std_stat_PET.statistic,
                             mean_stat_PET.statistic + std_stat_PET.statistic, facecolor='tab:orange', alpha=0.25)
            axes3.plot(mean_stat_P.statistic, bin_medians, color='tab:blue')
            axes3.fill_betweenx(bin_medians, mean_stat_P.statistic - std_stat_P.statistic,
                             mean_stat_P.statistic + std_stat_P.statistic, facecolor='tab:blue', alpha=0.25)

            axes2.plot(np.ones_like(np.linspace(0,bin_edges[-1],10)), np.linspace(0,bin_edges[-1],10), '--', c='gray', linewidth=0.5)
            axes2.set_xlabel('PET/P [-]')
            axes2.set_ylabel('Elevation [m]')
            axes2.set_xlim([0.1, 10])
            axes2.set_xscale('log')
            axes2.set_xticks(ticks=[0.2, 0.5, 1, 2, 5], labels=["0.2", "0.5", "1", "2", "5"])
            axes2.set_ylim([0, 6000])
            # plt.show()
            fig2.savefig(results_path + name + "/swaths_elevation_profiles/" + "swaths_elevation_profiles_" + name + "_" + str(p) + ".png", dpi=600, bbox_inches='tight')
            plt.close(fig2)

            axes3.set_xlabel('Flux [mm/y]')
            axes3.set_ylabel('Elevation [m]')
            axes3.set_xlim(forcinglim)
            axes3.set_ylim([0, 6000])
            # plt.show()
            axes3.grid(linewidth=0.5,color="lightgrey")
            fig3.savefig(results_path + name + "/swaths_elevation_profiles/" + "swaths_elevation_profiles_PET_P_" + name + "_" + str(p) + ".png", dpi=600, bbox_inches='tight')
            plt.close(fig3)

            # print gradients
            print(" ")
            reg = LinearRegression().fit(bin_medians.reshape(-1, 1),mean_stat_PET.statistic.reshape(-1, 1))
            print("PET lapse rate "+str(reg.coef_))
            reg = LinearRegression().fit(bin_medians.reshape(-1, 1),mean_stat_P.statistic.reshape(-1, 1))
            print("P lapse rate " + str(reg.coef_))
            print(" ")

    # plt.show()
    fig.savefig(results_path + name + "/swaths_elevation_profiles/" + "swaths_" + name + ".png", dpi=600, bbox_inches='tight')
    plt.close(fig)
