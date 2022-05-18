import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import mapping
import rioxarray as rxr
#import pyosp
import fiona
from matplotlib.pyplot import cm
from scipy import stats

# specify paths
data_path = r"D:/Data/"
data_path = "/home/hydrosys/data/"
results_path = "results/"

"""
pet_path_worldclim = data_path + "WorldClim/7504448/global-et0_annual.tif/et0_yr/et0_yr.tif"
pet_path_chelsa = data_path + "CHELSA/CHELSA_pet_penman_mean_1981-2010_V.2.1.tif"

pet_worldclim = rxr.open_rasterio(pet_path_worldclim, masked=True).squeeze() #
df_worldclim = pd.DataFrame()
X,Y = np.meshgrid(pet_worldclim.x,pet_worldclim.y)
df_worldclim["lon"] = X.flatten()
df_worldclim["lat"] = Y.flatten()
df_worldclim["pet_worldclim"] = pet_worldclim.values.flatten()
df_worldclim = df_worldclim.dropna()

df_worldclim.to_csv(results_path + 'worldclim.csv', index=False)

pet_chelsa = rxr.open_rasterio(pet_path_chelsa, masked=True).squeeze() #
df_chelsa = pd.DataFrame()
X,Y = np.meshgrid(pet_chelsa.x,pet_chelsa.y)
df_chelsa["lon"] = X.flatten()
df_chelsa["lat"] = Y.flatten()
df_chelsa["pet_chelsa"] = pet_chelsa.values.flatten()
df_chelsa = df_chelsa.dropna()
df_chelsa["pet_chelsa"] = df_chelsa["pet_chelsa"]*0.01*12

df_chelsa.to_csv(results_path + 'chelsa.csv', index=False)

#df_merged = pd.merge(df_worldclim, df_chelsa, on=['lat', 'lon'], how='outer')
#df_merged = df_merged.dropna()

#df_merged.to_csv(results_path + 'pet_data.csv', index=False)

print("Finished loading data.")
"""

def latitude_plots_data(df1, df2):

    # check if folder exists
    results_path = "results/latitude/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # make plot
    # specify names, units, and axes limits
    ax = plt.gca()
    ax.set_xlim([-100, 2900])
    ax.set_ylim([-60, 90])
    ax.set_xlabel('PET [mm/y]')
    ax.set_ylabel('lat [deg]')

    # calculate averages
    from functions.avg_group import mean_group
    latavg, chelsaavg = mean_group(df1["lat"].values, df1["pet_chelsa"].values)
    latavg, worldclimavg = mean_group(df2["lat"].values, df2["pet_worldclim"].values)
    print("Finished with averaging.")

    # plot latitudinal averages
    ax.plot(chelsaavg, latavg, c='tab:orange', label='CHELSA', alpha=0.8) #, mfc=nextcolor
    ax.plot(worldclimavg, latavg, c='tab:red', label='WORLDCLIM', alpha=0.8) #, mfc=nextcolor
    ax.legend()

    #plt.show()
    ax.savefig(results_path + 'pet_data' + "_latitude.png", dpi=600, bbox_inches='tight')
    plt.close()

    print("Finished latitude plots.")

#df = pd.read_csv(data_path + "pet_data.csv", sep=',')
df_chelsa = pd.read_csv(results_path + 'chelsa.csv', sep=',')
df_worldclim = pd.read_csv(results_path + 'worldclim.csv', sep=',')
latitude_plots_data(df_chelsa,df_worldclim)

