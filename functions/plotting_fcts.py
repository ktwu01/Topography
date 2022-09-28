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

def plot_origin_line(x, y, **kwargs):

    ax = plt.gca()
    lower_lim = min([ax.get_xlim()[0], ax.get_ylim()[0]])
    upper_lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot(np.linspace(lower_lim, upper_lim, 1000), np.linspace(lower_lim,  upper_lim, 1000), '--', color='black', alpha=0.5, zorder=1)


def plot_Budyko_limits(x, y, **kwargs):

    ax = plt.gca()
    lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', c='gray')
    ax.plot(np.linspace(1, lim, 100), np.linspace(1, 1, 100), '--', c='gray')


def plot_bins_group(x, y, color="tab:blue", group_type="aridity_class", group="energy-limited", **kwargs):

    # extract data
    df = kwargs.get('data')

    # get correlations
    df = df.dropna()
    df_group = df.loc[df[group_type]==group]

    # calculate binned statistics
    bin_edges, \
    mean_stat, std_stat, median_stat, \
    p_05_stat, p_25_stat, p_75_stat, p_95_stat, \
    asymmetric_error, bin_median = get_binned_stats(df_group[x], df_group[y])

    ax = plt.gca()
    corr_str = ''
    r_sp, _ = stats.spearmanr(df.loc[df[group_type] == group, x], df.loc[df[group_type] == group, y], nan_policy='omit')
    #corr_str = corr_str + r' $\rho_s$ ' + str(group) + " = " + str(np.round(r_sp,2))
    corr_str = corr_str + str(np.round(r_sp,2))
    print(corr_str)
    r_sp_tot, _ = stats.spearmanr(df[x], df[y], nan_policy='omit')
    print("(" + str(np.round(r_sp_tot,2)) + ")")

    # plot bins
    ax = plt.gca()
    ax.errorbar(bin_median, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
                fmt='o', ms=4, elinewidth=1, c='black', ecolor='black', mec='black', mfc=color, alpha=0.9, label=corr_str)


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
    #ax.plot(bin_median, median_stat.statistic, c=color, alpha=0.5, linewidth=1.0)

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


def add_corr(x, y, **kws):
    r_sp, _ = stats.spearmanr(x, y, nan_policy='omit')
    ax = plt.gca()
    ax.annotate("Sr: {:.2f}".format(r_sp), xy=(.1, .9), xycoords=ax.transAxes, fontsize=10)


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
