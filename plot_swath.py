import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import pyosp
from functions.get_geometries import get_swath_geometries
from functions.create_shapefiles import create_line_shp
from functions.create_shapefiles import create_polygon_shp
import os
from functions.get_swath_data import get_swath_data

# specify paths
#data_path = r"C:/Users/Sebastian/Documents/Data/"
data_path = r"D:/Data/"
results_path = "results/"

shp_path = data_path + "GMBA mountain inventory V1.2(entire world)/GMBA Mountain Inventory_v1.2-World.shp"
dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
pr_path = data_path + "wc2.1_30s_bio/wc2.1_30s_bio_12.tif"
pet_path = data_path + "wc2.1_30s_vapr/wc2.1_30s_vapr_avg.tif"
vap_path = data_path + "wc2.1_30s_vapr/wc2.1_30s_vapr_avg.tif"
t_path = data_path + "wc2.1_30s_bio/wc2.1_30s_bio_1.tif"

name_list = ["Cordillera Central Ecuador"]
#name_list = ["Cordillera principal"]
#["Sierra_Nevada", "European Alps", "Cordillera Central Ecuador", "France", "Himalaya", "Northern Alps", "Kilimanjaro", "Cascade Range"]

# load dem shapefile
dem = rxr.open_rasterio(dem_path, masked=True).squeeze() #todo: remove masked ...

# loop over mountain ranges
for name in name_list:

    # check if folder exists
    path = results_path + name + '/shapefiles/'
    if not os.path.isdir(path):
        os.makedirs(path)

    # load coordinates for lines
    xy_line, xy_box = get_swath_geometries(name)

    # create line
    baseline = results_path + name + '/shapefiles/line.shp'
    create_line_shp(xy_line, baseline)
    line_shape = pyosp.read_shape(baseline)
    lx, ly = line_shape.xy

    # generate swath objects
    w = 2 # 2 degrees + 2*cs
    ls = w/100
    cs = w/10
    orig_dem = pyosp.Orig_curv(baseline, dem_path, width=w, line_stepsize=ls, cross_stepsize=cs)
    orig_pr = pyosp.Orig_curv(baseline, pr_path, width=w, line_stepsize=ls, cross_stepsize=cs)
    orig_pet = pyosp.Orig_curv(baseline, pet_path, width=w, line_stepsize=ls, cross_stepsize=cs)
    orig_vap = pyosp.Orig_curv(baseline, vap_path, width=w, line_stepsize=ls, cross_stepsize=cs)
    orig_t = pyosp.Orig_curv(baseline, t_path, width=w, line_stepsize=ls, cross_stepsize=cs)

    ### PLOT 1 ###
    # initialise plot with 3 panels
    fig = plt.figure(figsize=(12, 3), constrained_layout=True)
    gs = plt.GridSpec(1, 3, figure=fig)
    axes0 = fig.add_subplot(gs[0, 0])
    axes1 = fig.add_subplot(gs[0, 1])
    axes2 = fig.add_subplot(gs[0, 2])
    #axes0.set_aspect('equal', adjustable='box')

    # plot swath lines and polygons
    swath_polylines = orig_dem.out_polylines()
    #for line in swath_polylines:
    #    x, y = line.xy
    #    axes0.plot(x, y, color='C2')

    swath_polygon = orig_dem.out_polygon()
    px, py = swath_polygon.exterior.xy
    axes0.plot(px, py, c='tab:orange')

    # save polygon as shapefile
    create_polygon_shp(swath_polygon, results_path + name + '/shapefiles/polygon.shp')

    #axes0.plot(lx, ly, color='C3', label="Baseline")
    #axes0.set_title("Swath profile lines")
    #axes0.legend()

    sp0 = dem.plot.imshow(ax=axes0, cmap='terrain')
    axes0.set(title=None) #"DEM [m]"
    #axes1.set_axis_off()
    axes0.axis('equal')
    axes0.set_xlim([xy_box[0], xy_box[1]])
    axes0.set_ylim([xy_box[2], xy_box[3]])
    axes0.set_xlabel('Lon [deg]')
    axes0.set_ylabel('Lat [deg]')
    sp0.colorbar.set_label('DEM [m]')
    sp0.set_clim([0, 100*round(np.nanmax(dem.values/100))])

    # plot swath profile
    dist, dem_swath, pr_swath, pet_swath, t_swath = \
        get_swath_data(orig_dem, orig_pr, orig_pet, orig_t, line_shape) #add orig_vap

    axes1.fill_between(dist, np.zeros(len(dist)), dem_swath.mean(axis=1),
                       facecolor='tab:gray', alpha=0.25, label='Elevation')
    axes1.fill_between(dist, np.zeros(len(dist)), dem_swath.mean(axis=1)-dem_swath.std(axis=1),
                       facecolor='tab:gray', alpha=0.25)
    axes1.fill_between(dist, np.zeros(len(dist)), dem_swath.mean(axis=1)+dem_swath.std(axis=1),
                       facecolor='tab:gray', alpha=0.25)
    #axes1.plot(dist, dem_swath.mean(axis=1), c='tab:grey', label='Elevation') #np.array(orig_dem.dat)[:,i]

    axes1b = axes1.twinx()
    axes1b.plot(dist, pr_swath.mean(axis=1),
               c='tab:blue', label='Precipitation') #np.array(orig_dem.dat)[:,i]
    axes1b.fill_between(dist, pr_swath.mean(axis=1)-pr_swath.std(axis=1), pr_swath.mean(axis=1)+pr_swath.std(axis=1),
                        facecolor='tab:blue', alpha=0.25)

    axes1b.plot(dist, pet_swath.mean(axis=1),
                c='tab:orange', label='PET')  # np.array(orig_dem.dat)[:,i]
    axes1b.fill_between(dist, pet_swath.mean(axis=1) - pet_swath.std(axis=1),
                        pet_swath.mean(axis=1) + pet_swath.std(axis=1),
                        facecolor='tab:orange', alpha=0.25)

    lines, labels = axes1.get_legend_handles_labels()
    lines2, labels2 = axes1b.get_legend_handles_labels()
    axes1b.legend(lines + lines2, labels + labels2)
    #axes1.legend().set_visible(False)
    #axes1.legend(loc='upper left')
    axes1.set_xlabel('Distance [deg]')
    axes1.set_ylabel('Elevation [m]')
    axes1b.set_ylabel('Flux [mm/y]')
    #axes1.set_ylim(0,5000) #todo: adjust limits
    #axes1b.set_ylim(0,5000)

    # plot elevation profile
    axes2.plot(pr_swath.mean(axis=1), dem_swath.mean(axis=1), c='grey', alpha=0.5)
    sp2 = axes2.scatter(pr_swath.mean(axis=1), dem_swath.mean(axis=1),
                 marker='o', c=dist)
    axes2.set_xlabel('Precipitation [mm/y]')
    axes2.set_ylabel('Elevation [m]')
    #axes2.set_xlim([0,5000])
    #axes2.set_ylim([0,5000])
    #axes2.set(title="Distance [deg]")
    cbar2 = plt.colorbar(sp2, ax=axes2)
    cbar2.ax.set_ylabel('Distance [deg]')
    #fig.tight_layout()

    #plt.show()
    plt.savefig(results_path + name + "/swath_" + name + ".png", dpi=600, bbox_inches='tight')
    plt.close()

    ### PLOT 2 ###
    # plot swath transect
    fig = plt.figure(figsize=(8, 2), constrained_layout=True)
    ax = plt.axes()

    axa = ax.twinx()
    axa.fill_between(dist, np.zeros(len(dist)), dem_swath.mean(axis=1),
                       facecolor='tab:gray', alpha=0.25, label='Elevation')
    axa.fill_between(dist, np.zeros(len(dist)), dem_swath.mean(axis=1)-dem_swath.std(axis=1),
                       facecolor='tab:gray', alpha=0.25)
    axa.fill_between(dist, np.zeros(len(dist)), dem_swath.mean(axis=1)+dem_swath.std(axis=1),
                       facecolor='tab:gray', alpha=0.25)
    #axes1.plot(dist, dem_swath.mean(axis=1), c='tab:grey', label='Elevation') #np.array(orig_dem.dat)[:,i]

    axb = ax.twinx()
    axb.plot(dist, pr_swath.mean(axis=1),
               c='tab:blue', label='Precipitation') #np.array(orig_dem.dat)[:,i]
    axb.fill_between(dist, pr_swath.mean(axis=1)-pr_swath.std(axis=1), pr_swath.mean(axis=1)+pr_swath.std(axis=1),
                        facecolor='tab:blue', alpha=0.25)

    axc = ax.twinx()
    """
    axc.plot(dist, vap_swath.mean(axis=1),
                c='tab:green', label='Vapor pressure')  # np.array(orig_dem.dat)[:,i]
    axc.fill_between(dist, vap_swath.mean(axis=1) - vap_swath.std(axis=1),
                        vap_swath.mean(axis=1) + vap_swath.std(axis=1),
                        facecolor='tab:green', alpha=0.25)
                        """
    axc.plot(dist, pet_swath.mean(axis=1),
                c='tab:orange', label='Potential evapotranspiration')  # np.array(orig_dem.dat)[:,i]
    axc.fill_between(dist, pet_swath.mean(axis=1) - pet_swath.std(axis=1),
                        pet_swath.mean(axis=1) + pet_swath.std(axis=1),
                        facecolor='tab:orange', alpha=0.25)


    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axes1b.get_legend_handles_labels()
    #axb.legend(lines + lines2, labels + labels2)
    #axes1.legend().set_visible(False)
    #axes1.legend(loc='upper left')
    #ax.set_xlabel('Distance [deg]')
    #axa.set_ylabel('Elevation [m]')
    #axb.set_ylabel('Precipitation [mm/y]')
    #axc.set_ylabel('Vapor pressure [Pa]')
    #axb.set_ylabel('Precipitation [mm/y] / Vapor pressure [Pa]')
    axa.set_ylim(0,5000)
    axb.set_ylim(0,5000)
    axc.set_ylim(0,5000)
    #axa.set_ylim(0,2500)
    #axb.set_ylim(0,2500)
    #axc.set_ylim(0,2500)
    axa.set_xlim([line_shape.xy[0][0], line_shape.xy[0][1]]) # works only for east-west swaths
    axb.set_xlim([line_shape.xy[0][0], line_shape.xy[0][1]]) # works only for east-west swaths
    axc.set_xlim([line_shape.xy[0][0], line_shape.xy[0][1]]) # works only for east-west swaths

    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])

    axa.spines['right'].set_position(('outward', 10))
    axa.spines.left.set_visible(False)
    axa.spines.top.set_visible(False)
    #axa.spines.bottom.set_visible(False)
    axa.spines['bottom'].set_position(('outward', 10))
    #axa.yaxis.label.set_color('tab:gray')
    #axa.tick_params(axis='y', colors='tab:gray')
    #axa.spines['right'].set_color('tab:gray')

    axb.spines['right'].set_position(('outward', 50))
    axb.yaxis.label.set_color('tab:blue')
    axb.tick_params(axis='y', colors='tab:blue')
    axb.spines['right'].set_color('tab:blue')
    axb.spines.left.set_visible(False)
    axb.spines.top.set_visible(False)
    axb.spines.bottom.set_visible(False)

    axc.spines['right'].set_position(('outward', 90))
    axc.yaxis.label.set_color('tab:orange')
    axc.tick_params(axis='y', colors='tab:orange')
    axc.spines['right'].set_color('tab:orange')
    axc.spines.left.set_visible(False)
    axc.spines.top.set_visible(False)
    axc.spines.bottom.set_visible(False)

    """
    # move left and bottom spines outward by 10 points
    ax.spines.left.set_position(('outward', 10))
    ax.spines.bottom.set_position(('outward', 10))
    axb.spines.right.set_position(('outward', 10))
    axb.spines.bottom.set_position(('outward', 10))
    # hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_xticklabels([])
    ax.set_xticks([])
    axb.spines.left.set_visible(False)
    axb.spines.top.set_visible(False)
    axb.spines.bottom.set_visible(False)
    axb.set_xticklabels([])
    axb.set_xticks([])
    """
    #plt.show()
    plt.savefig(results_path + name + "/transect_" + name + ".png", dpi=600, bbox_inches='tight')
    plt.close()

    ### PLOT 3 ###
    # plot T profile
    #todo: plot profile using grid cells
    fig3 = plt.figure(figsize=(2, 4), constrained_layout=True)
    axes3 = plt.axes()
    axes3.plot(t_swath.mean(axis=1), dem_swath.mean(axis=1), color="tab:purple", alpha=0.75)
    axes3.fill_betweenx(dem_swath.mean(axis=1), t_swath.mean(axis=1) - t_swath.std(axis=1),
                     t_swath.mean(axis=1) + t_swath.std(axis=1),
                     facecolor='tab:purple', alpha=0.25)
    axes3.set_ylabel('Elevation [m]')
    axes3.set_xlabel('T [°C]')
    axes3.set_xlim(-20,20)
    axes3.set_ylim(1000,7000) #todo: adjust limits

    # move left and bottom spines outward by 10 points
    axes3.spines.left.set_position(('outward', 10))
    axes3.spines.bottom.set_position(('outward', 10))
    # hide the right and top spines
    axes3.spines.right.set_visible(False)
    axes3.spines.top.set_visible(False)
    #axes3.set_xticklabels([])
    #axes3.set_xticks([])
    #axes3.grid()

    x_line = np.linspace(-20,20,2)
    axes3.fill_between(x_line, 5000*np.ones_like(x_line), 5500*np.ones_like(x_line),
                        facecolor='tab:gray', alpha=0.25, label='Snowline')

    axes3.fill_between(x_line, 3500*np.ones_like(x_line), 4000*np.ones_like(x_line),
                        facecolor='tab:green', alpha=0.25, label='Treeline')

    #plt.grid()

    #plt.show()
    plt.savefig(results_path + name + "/T_profile_" + name + ".png", dpi=600, bbox_inches='tight')
    plt.close()
