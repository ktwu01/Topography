import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as ml_colors
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from brewer2mpl import brewer2mpl
import os
from functools import reduce

# This script contains a collection of function that make or help with making plots.

### plots for each model ###

def plot_origin_line(x, y, **kwargs):

    ax = plt.gca()
    #lim = max([x.max(), y.max()])
    lower_lim = min([ax.get_xlim()[0], ax.get_ylim()[0]])
    upper_lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot(np.linspace(lower_lim, upper_lim, 1000), np.linspace(lower_lim,  upper_lim, 1000), '--', color='black', alpha=0.5, zorder=1)


def plot_Budyko_limits(x, y, **kwargs):

    ax = plt.gca()
    #lim = max([x.max(), y.max()])
    lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', c='gray')
    ax.plot(np.linspace(1, lim, 100), np.linspace(1, 1, 100), '--', c='gray')


def plot_bins(x, y, fillcolor='black', **kwargs):

    # calculate binned statistics
    bin_edges, \
    mean_stat, std_stat, median_stat, \
    p_05_stat, p_25_stat, p_75_stat, p_95_stat, \
    asymmetric_error, bin_median = get_binned_stats(x, y)

    # plot bins
    ax = plt.gca()
    color = 'black'
    ax.errorbar(bin_median, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
                fmt='o', ms=4, elinewidth=1, c=color, ecolor=color, mec=color, mfc=fillcolor, alpha=0.9)
    ax.plot(bin_median, median_stat.statistic, c=color, alpha=0.5, linewidth=1.0)

    lim=ax.get_ylim()[1]
    ax.plot(np.linspace(np.min(x), np.max(x), 1000), 0.9 * lim * np.ones(1000),
            '-', c='black', alpha=0.2, linewidth=8, solid_capstyle='butt')
    ax.errorbar(bin_edges[1:], 0.9*lim*np.ones(10), xerr=None, yerr=0.025*lim*np.ones(10),
                c='white', zorder=10, fmt='none')


def binned_stats_table(df, x_str, y_str, ghms):

    l = []
    for g in ghms:
        x = df.loc[df["ghm"]==g, x_str]
        y = df.loc[df["ghm"]==g, y_str]

        # calculate binned statistics
        bin_edges, \
        mean_stat, std_stat, median_stat, \
        p_05_stat, p_25_stat, p_75_stat, p_95_stat, \
        asymmetric_error, bin_median = get_binned_stats(x, y)

        results = pd.DataFrame()
        results["bin_lower_edge"] = bin_edges[0:-1]
        results["bin_upper_edge"] = bin_edges[1:]
        results["bin_median"] = bin_median
        results["mean"] = mean_stat.statistic
        results["std"] = std_stat.statistic
        results["median"] = median_stat.statistic
        results["05_perc"] = p_05_stat.statistic
        results["25_perc"] = p_25_stat.statistic
        results["75_perc"] = p_75_stat.statistic
        results["95_perc"] = p_95_stat.statistic
        results["ghm"] = g

        l.append(results)

    results_df = pd.concat(l)

    return results_df


def get_binned_stats(x, y):

    # calculate binned statistics
    bin_edges = stats.mstats.mquantiles(x[~np.isnan(x)], np.linspace(0, 1, 11))
    #bin_edges = np.linspace(-.5,10.5,12)
    mean_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmean(y), bins=bin_edges)
    std_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanstd(y), bins=bin_edges)
    median_stat = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=bin_edges)
    p_05_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .05), bins=bin_edges)
    p_25_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .25), bins=bin_edges)
    p_75_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .75), bins=bin_edges)
    p_95_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .95), bins=bin_edges)
    asymmetric_error = [median_stat.statistic - p_25_stat.statistic, p_75_stat.statistic - median_stat.statistic]
    bin_median = stats.mstats.mquantiles(x, np.linspace(0.05, 0.95, 10))

    return bin_edges, \
           mean_stat, std_stat, median_stat, \
           p_05_stat, p_25_stat, p_75_stat, p_95_stat, \
           asymmetric_error, bin_median


"""
def plot_coloured_scatter(x, y, z, **kws):

    ax = plt.gca()
    sp = ax.scatter(x, y, marker='o', s=.5, c=z, cmap='viridis', alpha=0.1) #, vmin=-200, vmax=1800
    cbar = plt.colorbar(sp, ax=ax)
    cbar.solids.set(alpha=1)
    #cbar.set_label(z_name + z_unit, size=4)
    cbar.ax.tick_params(labelsize=4)
"""


def add_corr(x, y, **kws):
    #r_lin, _ = stats.pearsonr(x, y, nan_policy='omit')
    r_sp, _ = stats.spearmanr(x, y, nan_policy='omit')
    ax = plt.gca()
    #ax.annotate("Pr: {:.2f}".format(r_lin), xy=(.3, .45), xycoords=ax.transAxes, fontsize=10)
    ax.annotate("Sr: {:.2f}".format(r_sp), xy=(.1, .9), xycoords=ax.transAxes, fontsize=10)


def plot_latitudinal_averages(xlim=[-100, 2400], x_name='flux', x_unit=' [mm/year]', **kwargs):

    df = kwargs["data"]
    # specify names, units, and axes limits
    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim([-60, 90])
    ax.set_xlabel(x_name + x_unit)
    ax.set_ylabel('lat [deg]')

    # calculate averages
    from lib.avg_group import mean_group
    latavg, pravg = mean_group(df["lat"].values, df["pr"].values)
    latavg, qravg = mean_group(df["lat"].values, df["qr"].values)
    latavg, evapavg = mean_group(df["lat"].values, df["evap"].values)
    latavg, qtotavg = mean_group(df["lat"].values, df["qtot"].values)

    # plot latitudinal averages
    # water balance 1
    #ax.fill_betweenx(latavg, -evapavg, pravg-evapavg, facecolor='grey', label='pr', alpha=0.5) #, mfc=nextcolor
    #ax.fill_betweenx(latavg, qtotavg, facecolor='navy', label='qtot', alpha=0.5) #, mfc=nextcolor
    #ax.fill_betweenx(latavg, -evapavg, facecolor='green', label='evap', alpha=0.5) #, mfc=nextcolor

    # water balance 2
    ax.fill_betweenx(latavg, qtotavg+evapavg, qtotavg, facecolor='green', label='evap', alpha=0.8) #, mfc=nextcolor
    ax.fill_betweenx(latavg, qtotavg, facecolor='navy', label='qtot', alpha=0.8) #, mfc=nextcolor
    ax.plot(pravg, latavg, c='dimgray', label='pr', alpha=0.8) #, mfc=nextcolor

    # flow components
    #ax.plot(qravg, latavg, c='skyblue', label='qr', alpha=0.8) #, mfc=nextcolor
    #ax.plot(qtotavg, latavg, '-', c='navy', label='qtot', alpha=0.8) #, mfc=nextcolor
    #ax.plot(pravg, latavg, c='dimgray', label='pr', alpha=0.8) #, mfc=nextcolor

    # all combined
    #ax.fill_betweenx(latavg, qtotavg, facecolor='navy', label='qtot', alpha=0.8) #, mfc=nextcolor
    #ax.fill_betweenx(latavg, qtotavg, qtotavg+evapavg, facecolor='green', label='evap', alpha=0.5) #, mfc=nextcolor
    #ax.plot(qravg, latavg, '-', c='skyblue', label='qr', alpha=0.8) #, mfc=nextcolor
    #ax.plot(pravg, latavg, c='dimgray', label='pr', alpha=0.8) #, mfc=nextcolor

    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)


### plots that create maps ###

def mask_greenland(data_path="2b/aggregated/"):
    ax = plt.gca()
    df_greenland = pd.read_csv(data_path + "greenland.csv", sep=',')  # greenland mask for plot
    ax.scatter(df_greenland['lon'], df_greenland['lat'], transform=ccrs.PlateCarree(),
               marker='s', s=.35, edgecolors='none', c='lightgray')


def plot_map(lon, lat, var, var_unit=" ", var_name="misc",
             bounds=np.linspace(0, 2000, 11), colormap='YlGnBu', colormap_reverse=False):

    # prepare colour map
    o = brewer2mpl.get_map(colormap, 'Diverging', 9, reverse=colormap_reverse)
    c = o.mpl_colormap

    # create figure
    plt.rcParams['axes.linewidth'] = 0.1
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    customnorm = ml_colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    sc = ax.scatter(lon, lat, c=var, cmap=c, marker='s', s=.35, edgecolors='none',
                    norm=customnorm, transform=ccrs.PlateCarree())
    #ax.coastlines(linewidth=0.5)

    box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
    x0, y0, x1, y1 = box.bounds
    ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())

    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.5)
    cbar.set_label(var_name + var_unit)
    # cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
    cbar.ax.tick_params(labelsize=6)
    plt.gca().outline_patch.set_visible(False)


def plot_outliers(df, g, var_name, var_unit=" "):

    # create figure
    plt.rcParams['axes.linewidth'] = 0.1
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    df_m = df[df["ghm"] == g]

    sc = ax.scatter(df_m['lon'].mask(df_m[var_name]<=df_m["pr"]), df_m['lat'].mask(df_m[var_name]<=df_m["pr"]),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='var>precip')
    sc = ax.scatter(df_m['lon'].mask(df_m[var_name] < 10000), df_m['lat'].mask(df_m[var_name] < 10000),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='var>10000 (large)')
    sc = ax.scatter(df_m['lon'].mask((df_m[var_name] > 1) | (df_m[var_name] <= 0)), df_m['lat'].mask((df_m[var_name] > 1) | (df_m[var_name] <= 0)),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='0<var<=1 (small)')
    sc = ax.scatter(df_m['lon'].mask(df_m[var_name] != 0), df_m['lat'].mask(df_m[var_name] != 0),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='var=0 (zero)')
    sc = ax.scatter(df_m['lon'].mask((df_m[var_name] >= 0)), df_m['lat'].mask((df_m[var_name] >= 0)),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='var<0 (negative)')
    df_m[var_name][np.isnan(df_m[var_name])] = -999 # todo: change this line to remove warning
    sc = ax.scatter(df_m['lon'].mask(df_m[var_name] != -999), df_m['lat'].mask(df_m[var_name] != -999),
                    transform=ccrs.PlateCarree(), marker='s', s=.35, edgecolors='none', label='var=nan',
                    facecolor='tab:grey')
    # ax.coastlines(linewidth=0.5)

    box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
    x0, y0, x1, y1 = box.bounds
    ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())

    # cbar = plt.colorbar(sc,orientation='horizontal', pad=0.01, shrink=.5)
    # cbar.set_label(var_name + var_unit)
    # cbar.set_ticks([1,0,-1,-np.nan])
    # cbar.ax.tick_params(labelsize=6)
    plt.gca().outline_patch.set_visible(False)

    ax.set_title(g, fontsize=10)
    leg = ax.legend(loc='lower left', markerscale=4, fontsize=7)
    leg.set_title(title="var = " + var_name + var_unit, prop={'size': 8})


def plot_most_deviating_model(df, max_ind, ghms, var_name):

    o = brewer2mpl.get_map("Dark2", 'Qualitative', 8, reverse=True)
    c = o.mpl_colormap

    # create figure
    plt.rcParams['axes.linewidth'] = 0.1
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    bounds = np.linspace(-1.5, len(ghms) - 0.5, len(ghms) + 2)
    customnorm = ml_colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    sc = ax.scatter(df['lon'], df['lat'], transform=ccrs.PlateCarree(), norm=customnorm,
                    marker='s', s=.35, edgecolors='none', c=np.ma.masked_equal(max_ind, -2), cmap=c)

    box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
    x0, y0, x1, y1 = box.bounds
    ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())

    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.5)
    cbar.set_label(var_name + ' [-]')
    # cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
    cbar.ax.tick_params(labelsize=3)
    plt.gca().outline_patch.set_visible(False)

    cbar.set_ticks(np.arange(-1, len(ghms), 1))
    cbar.set_ticklabels(["multiple"] + ghms)
