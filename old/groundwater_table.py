import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio

# specify paths
data_path = "/home/hydrosys/data/" #r"C:/Users/Sebastian/Documents/Data/"
#data_path = r"D:/Data/"
results_path = "results/"

dem_path = data_path + "WorldClim/" + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
slope_path = data_path + "DEMs/Geomorpho90m/" + "dtm_slope_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
conv_path = data_path + "DEMs/Geomorpho90m/" + "dtm_convergence_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
twi_path = data_path + "DEMs/Geomorpho90m/" + "dtm_cti_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
geom_path = data_path + "DEMs/Geomorpho90m/" + "dtm_geom_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
pr_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_12.tif"
pet_path = data_path + "WorldClim/7504448/global-et0_annual.tif/et0_yr/et0_yr.tif"
t_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_1.tif"
wtd_model_path = data_path + "WorldClim/wtd.tif"
WTD_all_path = data_path + "Groundwater/WTD_all/water_table_depth_all_models.csv"

Fan_data_path = "Groundwater/Fan_2013_WTD/"
wtd_path_list = [Fan_data_path + "All-Africa-Data-lat-lon-z-wtd/All-Africa-Data-lat-lon-z-wtd.txt",
                 Fan_data_path + "All-Asia-Data-lat-lon-z-wtd/All-Asia-Data-lat-lon-z-wtd.txt",
                 Fan_data_path + "All-Australia-Data-lat-lon-z-wtd/All-Australia-Data-lat-lon-z-wtd.txt",
                 Fan_data_path + "All-Canada-Data-lat-lon-z-wtd/All-Canada-Data-lat-lon-z-wtd.txt",
                 Fan_data_path + "All-Europe-Data-lat-lon-z-wtd/All-Europe-Data-lat-lon-z-wtd.txt",
                 Fan_data_path + "All-S-America-Data-lat-lon-z-wtd/All-S-America-Data-lat-lon-z-wtd.txt",
                 Fan_data_path + "All-US-Data-lat-lon-z-wtd/All-US-Data-lat-lon-z-wtd.txt"]
"""
wtd_path_list = [Fan_data_path + "All-Africa-Data-lat-lon-z-wtd/All-Africa-Data-lat-lon-z-wtd.txt",
                 Fan_data_path + "All-Asia-Data-lat-lon-z-wtd/All-Asia-Data-lat-lon-z-wtd.txt",
                 Fan_data_path + "All-Australia-Data-lat-lon-z-wtd/All-Australia-Data-lat-lon-z-wtd.txt"]
"""
wtd_path_list_full = [data_path + s for s in wtd_path_list]

# open raster
#dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
dem = rio.open(dem_path, masked=True)
slope = rio.open(slope_path, masked=True)
conv = rio.open(conv_path, masked=True)
twi = rio.open(twi_path, masked=True)
geom = rio.open(geom_path, masked=True)
pr = rio.open(pr_path, masked=True)
pet = rio.open(pet_path, masked=True)
t = rio.open(t_path, masked=True)
wtd_model = rio.open(wtd_model_path, masked=True)

wtd_all = pd.read_csv(WTD_all_path, sep=',')

# load wtd data
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

# todo: check with Robert how many data points there are
df.loc[df['wtd'] > 9999, 'wtd'] = np.nan
# open shapefile and plot
#mountain_shp = gpd.read_file(shp_path)

# plot histogram
bins = np.linspace(-1,100,102)
h, bins = np.histogram(df['wtd'], bins=bins)
f, ax = plt.subplots(figsize=(6, 3))
#sp = landforms.plot.hist(ax=ax, bins=bins)
sp = ax.bar((bins[0:-1]+bins[1:])/2, h)
ax.set_xlabel('WTD [m]')
ax.set_ylabel('Count')
plt.savefig(results_path + "wtd_histogram.png", dpi=600, bbox_inches='tight')
plt.close()

# create geodataframe
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
# todo: ask Robert about link between df and gdf

# todo: CLEAN UP CODE - first do proper saving of data, then proper saving of general dataframe, then use dataframe to play around with...
"""
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
ax = plt.axes()#projection=ccrs.Robinson()
sp0 = dem.plot.imshow(ax=ax, cmap='Greys')
ax.set(title=None)  # "DEM [m]"
# axes.set_axis_off()
# axes.axis('equal')
ax.set_xlim([-180, 180])
ax.set_ylim([-60, 90])
ax.set_xlabel('Lon [deg]')
ax.set_ylabel('Lat [deg]')
sp0.colorbar.set_label('DEM [m]')
# sp0.set_clim([0, np.round(np.array(orig_dem.dat).max(), 100)])
sp0.set_clim([0, 5000])  # 100*round(np.max(dem.values/100))

gdf.plot("wtd", ax=ax, markersize=0.01, cmap="YlGnBu", vmin=0, vmax=50)

plt.savefig(results_path + "wtd_map.png", dpi=600, bbox_inches='tight')
plt.close()
"""

# extract point values from gridded files / shapefile
coord_list = [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]
gdf['dem'] = [x for x in dem.sample(coord_list)]
gdf['slope'] = [x for x in slope.sample(coord_list)]
gdf['conv'] = [x for x in conv.sample(coord_list)]
gdf['twi'] = [x for x in twi.sample(coord_list)]
gdf['geom'] = [x for x in geom.sample(coord_list)]
gdf['pr'] = [x for x in pr.sample(coord_list)]
gdf['pet'] = [x for x in pet.sample(coord_list)]
gdf['t'] = [x for x in t.sample(coord_list)]
gdf['wtd_model'] = [x for x in wtd_model.sample(coord_list)]
#gdf.head()
gdf['dem'] = np.concatenate(df['dem'].to_numpy())
gdf['slope'] = np.concatenate(df['slope'].to_numpy())
gdf['conv'] = np.concatenate(df['conv'].to_numpy())
gdf['twi'] = np.concatenate(df['twi'].to_numpy())
gdf['geom'] = np.concatenate(df['geom'].to_numpy())
gdf['pr'] = np.concatenate(df['pr'].to_numpy())
gdf['pet'] = np.concatenate(df['pet'].to_numpy())
gdf['t'] = np.concatenate(df['t'].to_numpy())
gdf['wtd_model'] = np.concatenate(df['wtd_model'].to_numpy())
gdf['aridity'] = gdf['pet']/gdf['pr']

gdf.to_file(results_path + 'dataframe.shp')

"""
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()#projection=ccrs.Robinson()
ax.scatter(gdf['aridity'], gdf['wtd'], s=5, facecolor='tab:blue', edgecolor='none', alpha=0.1)
plotting_fcts.plot_bins(gdf['aridity'], gdf['wtd'])
ax.set_xlabel('PET/P [-]')
#ax.set_xlabel('Elevation [m]')
ax.set_ylabel('WTD [m]')
ax.set_xlim([0.2, 50])
ax.set_xscale('log')
#ax.set_xlim([-100, 3900])
ax.set_ylim([0.1, 100])
ax.set_yscale('log')
idx = np.isfinite(gdf['aridity']) & np.isfinite(gdf['wtd'])
m, b = np.polyfit(gdf['aridity'][idx], gdf['wtd'][idx], 1)
#plt.plot(np.linspace(0,10000,10), m*np.linspace(0,10000,10) + b)
#plt.show()
#print(stats.spearmanr(gdf['value'], gdf['wtd'], nan_policy='omit'))
plt.savefig(results_path + "wtd_vs_aridity.png", dpi=600, bbox_inches='tight')
plt.close()
"""
