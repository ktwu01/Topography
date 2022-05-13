import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio
import plotting_fcts
from rasterio.crs import CRS
import rioxarray as rxr
import os
from scipy import stats

# specify paths
data_path = "/home/hydrosys/data/" #r"C:/Users/Sebastian/Documents/Data/"
#data_path = r"D:/Data/"
results_path = "results/WTD/"

if not os.path.isdir(results_path):
    os.makedirs(results_path)

def prepare_data(data_path, results_path):
    # specify file paths
    wtr_path = data_path + "Groundwater/Cuthbert_2019_WTR/" + "LOG_WTR_NL_01.tif"
    l_path = data_path + "Groundwater/Cuthbert_2019_WTR/" + "L01_m.tif"
    grt_path = data_path + "Groundwater/Cuthbert_2019_WTR/" + "Log_GRT_a.tif"

    Fan_data_path = "Groundwater/Fan_2013_WTD/"
    wtd_path_list = [Fan_data_path + "All-Africa-Data-lat-lon-z-wtd/All-Africa-Data-lat-lon-z-wtd.txt",
                     Fan_data_path + "All-Asia-Data-lat-lon-z-wtd/All-Asia-Data-lat-lon-z-wtd.txt",
                     Fan_data_path + "All-Australia-Data-lat-lon-z-wtd/All-Australia-Data-lat-lon-z-wtd.txt",
                     Fan_data_path + "All-Canada-Data-lat-lon-z-wtd/All-Canada-Data-lat-lon-z-wtd.txt",
                     Fan_data_path + "All-Europe-Data-lat-lon-z-wtd/All-Europe-Data-lat-lon-z-wtd.txt",
                     Fan_data_path + "All-S-America-Data-lat-lon-z-wtd/All-S-America-Data-lat-lon-z-wtd.txt",
                     Fan_data_path + "All-US-Data-lat-lon-z-wtd/All-US-Data-lat-lon-z-wtd.txt"]
    wtd_path_list_full = [data_path + s for s in wtd_path_list]


    # save wtd data in single file
    li = []
    for filename in wtd_path_list:
        df = pd.read_csv(data_path + filename, index_col=None, sep='\t')
        if 'Unnamed: 4' in df.columns: # extra tab after last column header in Africa file
            del df['Unnamed: 4']
        mapping = {df.columns[0]: 'lat', df.columns[1]: 'lon', df.columns[2]: 'z', df.columns[3]: 'wtd'}
        df = df.rename(columns=mapping)
        li.append(df)
    #frame = pd.concat(li, axis=1, ignore_index=True)
    df = pd.concat(li, axis=0)
    #df.loc[df['z'] ==" 1,005.00", 'z'] = np.nan
    #df['z'] = df['z'].str.replace(',','') # some values contain commas...
    #df.loc[df['wtd'] > 9999, 'wtd'] = np.nan
    df.to_csv(results_path + "WTD_Fan_2013.csv", index=False)

    # prepare raster datasets
    crs_wgs84 = CRS.from_string('EPSG:4326')
    # Cuthbert data
    wtr = rxr.open_rasterio(wtr_path, masked=True).squeeze()
    wtr = wtr.rio.reproject(crs_wgs84)
    # todo: file becomes much bigger after projection (and contains more elements...)
    wtr.rio.to_raster(results_path + "WTR_reprojected.tif")
    l = rxr.open_rasterio(l_path, masked=True).squeeze()
    l = l.rio.reproject(crs_wgs84)
    l.rio.to_raster(results_path + "L_reprojected.tif")
    grt = rxr.open_rasterio(grt_path, masked=True).squeeze()
    grt = grt.rio.reproject(crs_wgs84)
    grt.rio.to_raster(results_path + "GRT_reprojected.tif")

def load_data(data_path, results_path):
    # specify file paths
    dem_path = data_path + "WorldClim/" + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
    pr_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_12.tif"
    pet_path = data_path + "WorldClim/7504448/global-et0_annual.tif/et0_yr/et0_yr.tif"
    t_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_1.tif"
    wtd_model_path = data_path + "WorldClim/wtd.tif"

    slope_path = data_path + "DEMs/Geomorpho90m/" + "dtm_slope_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
    conv_path = data_path + "DEMs/Geomorpho90m/" + "dtm_convergence_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
    twi_path = data_path + "DEMs/Geomorpho90m/" + "dtm_cti_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
    geom_path = data_path + "DEMs/Geomorpho90m/" + "dtm_geom_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"

    wtr_path = results_path + "WTR_reprojected.tif"
    l_path = results_path + "L_reprojected.tif"
    grt_path = results_path + "GRT_reprojected.tif"

    wtd_path = results_path + "WTD_Fan_2013.csv"
    WTD_all_path = data_path + "Groundwater/WTD_all/water_table_depth_all_models.csv"

    # open all datasets
    df = pd.read_csv(results_path + "WTD_Fan_2013.csv", sep=',')
    dem = rio.open(dem_path, masked=True)
    slope = rio.open(slope_path, masked=True)
    conv = rio.open(conv_path, masked=True)
    twi = rio.open(twi_path, masked=True)
    geom = rio.open(geom_path, masked=True)
    wtr = rio.open(wtr_path, masked=True)
    l = rio.open(l_path, masked=True)
    grt = rio.open(grt_path, masked=True)
    pr = rio.open(pr_path, masked=True)
    pet = rio.open(pet_path, masked=True)
    t = rio.open(t_path, masked=True)
    wtd_model = rio.open(wtd_model_path, masked=True)

    #wtd_all = pd.read_csv(WTD_all_path, sep=',')

    # extract point values from gridded files / shapefile
    coord_list = [(x,y) for x,y in zip(df['lon'], df['lat'])]
    df['dem'] = [x for x in dem.sample(coord_list)]
    df['l'] = [x for x in l.sample(coord_list)]
    df['slope'] = [x for x in slope.sample(coord_list)]
    df['conv'] = [x for x in conv.sample(coord_list)]
    df['twi'] = [x for x in twi.sample(coord_list)]
    df['geom'] = [x for x in geom.sample(coord_list)]
    df['wtr'] = [x for x in wtr.sample(coord_list)]
    df['grt'] = [x for x in grt.sample(coord_list)]
    df['pr'] = [x for x in pr.sample(coord_list)]
    df['pet'] = [x for x in pet.sample(coord_list)]
    df['t'] = [x for x in t.sample(coord_list)]
    df['wtd_model'] = [x for x in wtd_model.sample(coord_list)]
    #gdf.head()
    df['dem'] = np.concatenate(df['dem'].to_numpy())
    df['l'] = np.concatenate(df['l'].to_numpy())
    df['slope'] = np.concatenate(df['slope'].to_numpy())
    df['conv'] = np.concatenate(df['conv'].to_numpy())
    df['twi'] = np.concatenate(df['twi'].to_numpy())
    df['geom'] = np.concatenate(df['geom'].to_numpy())
    df['wtr'] = np.concatenate(df['wtr'].to_numpy())
    df['grt'] = np.concatenate(df['grt'].to_numpy())
    df['pr'] = np.concatenate(df['pr'].to_numpy())
    df['pet'] = np.concatenate(df['pet'].to_numpy())
    df['t'] = np.concatenate(df['t'].to_numpy())
    df['wtd_model'] = np.concatenate(df['wtd_model'].to_numpy())
    df['aridity'] = df['pet']/df['pr']

    df.to_csv(results_path + 'wtd_df.csv', index=False)

def plot_data(data_path, results_path):
    # plot data
    df = pd.read_csv(results_path + 'wtd_df.csv')
    df.loc[df["aridity"] == np.inf, "aridity"] = np.nan
    df = df.dropna()

    # todo: clean up if statements...

    var_list =["dem", "slope", "conv", "aridity", "twi", "wtd_model", "l", "wtr", "grt"]
    #var_list =["l", "wtr", "grt"]
    for var in var_list:
        print(var)
        fig = plt.figure(figsize=(3, 3))
        ax = plt.axes()#projection=ccrs.Robinson()
        df.loc[df["aridity"] < 1, "wtd"]
        ax.scatter(df.loc[df["aridity"] > 1, var], df.loc[df["aridity"] > 1, "wtd"], s=1, facecolor='tab:orange', edgecolor='none', alpha=0.1)
        plotting_fcts.plot_bins(df.loc[df["aridity"] > 1, var], df.loc[df["aridity"] > 1, "wtd"], fillcolor='tab:orange')
        ax.scatter(df.loc[df["aridity"] < 1, var], df.loc[df["aridity"] < 1, "wtd"], s=1, facecolor='tab:blue', edgecolor='none', alpha=0.1)
        plotting_fcts.plot_bins(df.loc[df["aridity"] < 1, var], df.loc[df["aridity"] < 1, "wtd"], fillcolor='tab:blue')
        ax.set_xlabel(var)
        ax.set_ylabel('WTD [m]')
        if var == "conv":
            ax.set_xlim([np.nanquantile(df[var],0.01), np.nanquantile(df[var],0.99)])
        elif var == "geom":
            pass
        elif var == "twi":
            ax.set_xlim([np.nanquantile(df[var],0.01), np.nanquantile(df[var],0.99)])
        elif var == "wtd_model":
            ax.set_xscale('log')
            ax.set_xlim([.1, 100])
        elif var == "wtr":
            ax.set_xlim([-3, 3])
        else:
            ax.set_xscale('log')
            ax.set_xlim([np.nanquantile(df[var],0.01), np.nanquantile(df[var],0.99)])
        #ax.set_ylim([np.nanquantile(df["wtd"],0.01), np.nanquantile(df["wtd"],0.99)])
        ax.set_ylim([.1, 100])
        ax.set_yscale('log')
        #if var != "aridity":
        rho_s1, _ = stats.spearmanr(df.loc[df["aridity"] > 1, var], df.loc[df["aridity"] > 1, "wtd"], nan_policy='omit')
        rho_s2, _ = stats.spearmanr(df.loc[df["aridity"] < 1, var], df.loc[df["aridity"] < 1, "wtd"])
        ax.annotate("rho_s arid: {:.2f} ".format(rho_s1), xy=(.1, .9), xycoords=ax.transAxes, fontsize=10)
        ax.annotate("rho_s humid: {:.2f} ".format(rho_s2), xy=(.1, .83), xycoords=ax.transAxes, fontsize=10)
        plt.savefig(results_path + "wtd_vs_" + var + ".png", dpi=600, bbox_inches='tight')
        plt.close()

    var_list = ["dem", "slope", "conv", "aridity", "twi", "l", "wtr", "grt"]
    for var in var_list:
        print(var)
        fig = plt.figure(figsize=(3, 3))
        ax = plt.axes()#projection=ccrs.Robinson()
        df.loc[df["aridity"] < 1, "wtd"]
        ax.scatter(df.loc[df["aridity"] > 1, var], df.loc[df["aridity"] > 1, "wtd_model"], s=1, facecolor='tab:orange', edgecolor='none', alpha=0.1)
        plotting_fcts.plot_bins(df.loc[df["aridity"] > 1, var], df.loc[df["aridity"] > 1, "wtd_model"], fillcolor='tab:orange')
        ax.scatter(df.loc[df["aridity"] < 1, var], df.loc[df["aridity"] < 1, "wtd_model"], s=1, facecolor='tab:blue', edgecolor='none', alpha=0.1)
        plotting_fcts.plot_bins(df.loc[df["aridity"] < 1, var], df.loc[df["aridity"] < 1, "wtd_model"], fillcolor='tab:blue')
        ax.set_xlabel(var)
        ax.set_ylabel('WTD model [m]')
        if var == "conv":
            ax.set_xlim([np.nanquantile(df[var],0.01), np.nanquantile(df[var],0.99)])
        elif var == "geom":
            pass
        elif var == "twi":
            ax.set_xlim([np.nanquantile(df[var],0.01), np.nanquantile(df[var],0.99)])
        elif var =="wtd_model":
            ax.set_xscale('log')
            ax.set_xlim([.1, 100])
        elif var == "wtr":
            ax.set_xlim([-3, 3])
        else:
            ax.set_xscale('log')
            ax.set_xlim([np.nanquantile(df[var],0.01), np.nanquantile(df[var],0.99)])
        #ax.set_ylim([np.nanquantile(df["wtd"],0.01), np.nanquantile(df["wtd"],0.99)])
        ax.set_ylim([.1, 100])
        ax.set_yscale('log')
        #if var != "aridity":
        rho_s1, _ = stats.spearmanr(df.loc[df["aridity"] > 1, var], df.loc[df["aridity"] > 1, "wtd_model"], nan_policy='omit')
        rho_s2, _ = stats.spearmanr(df.loc[df["aridity"] < 1, var], df.loc[df["aridity"] < 1, "wtd_model"])
        ax.annotate("rho_s arid: {:.2f} ".format(rho_s1), xy=(.1, .9), xycoords=ax.transAxes, fontsize=10)
        ax.annotate("rho_s humid: {:.2f} ".format(rho_s2), xy=(.1, .83), xycoords=ax.transAxes, fontsize=10)
        plt.savefig(results_path + "wtd_model_vs_" + var + ".png", dpi=600, bbox_inches='tight')
        plt.close()


#prepare_data(data_path, results_path)
#load_data(data_path, results_path)
plot_data(data_path, results_path)

#df_obs = wtd_all.loc[wtd_all['Model'] == 'Fan et al. 2013 (observations)'] # ..
# todo: multiple point with same coordintes (90 lat... remove?)
# use Robert's dataframe
# = wtd_all.loc[wtd_all['Model'] == 'G³M']
#df['model_g3m'] = [x for x in wtd_tmp.sample(coord_list)]
#df_obs.columns = ["lon", "lat", "wtd_robert", "model"]
#df_test = pd.merge(df_obs, df, on=['lat', 'lon'], how='outer')

