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

# specify paths
data_path = r"C:/Users/Sebastian/Documents/Data/" #r"C:/Users/gnann/Documents/QGIS/Topography/"
results_path = "results/" #r"C:/Users/gnann/Documents/PYTHON/Topography/results/"

dem_path = data_path + "wc2.1_30s_elev/wc2.1_30s_elev.tif"

profile_path = data_path + "profile_cascades_45deg.csv"
name = "cascades_45deg"

# open raster and plot
dem = rxr.open_rasterio(dem_path, masked=True).squeeze()

# open profile
df = pd.read_csv(profile_path, sep=';')

# plot
fig, axes = plt.subplots(2, 1, figsize=(6, 6))

axes[0].fill_between(df['dist'], np.zeros(len(df['elev'])), df['elev'], facecolor='tab:gray', alpha=0.8)
axes[0].plot(df['dist'], df['pr']) # , c='tab:blue'
axes[0].set_ylabel('Elevation [m] / P [mm/year]')
axes[0].set_xlabel('Distance [deg]')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
#axs[i].spines['bottom'].set_visible(False)
#axs[i].spines['left'].set_visible(False)

#hillshade = es.hillshade(dem)
sp = dem.plot.imshow(cmap='gist_earth')
#axes[1].imshow(hillshade, cmap="Greys", alpha=0.5)
axes[1].plot(df['lon'], df['lat'], c='tab:orange')
axes[1].set(title="DEM [m]")
#axes[1].set_axis_off()
axes[1].axis('equal')
axes[1].set_xlim([df['lon'].iloc[0]-.1, df['lon'].iloc[-1]+.1])
axes[1].set_ylim([df['lat'].iloc[0]-3, df['lat'].iloc[-1]+3])
axes[1].set_xlabel('Lon [deg]')
axes[1].set_ylabel('Lat [deg]')
#cbar = plt.colorbar(sp, ax=axes[1])
sp.set_clim([0, 2000])

fig.tight_layout()

#plt.show()
plt.savefig(results_path + "transect_" + name + ".png", dpi=600, bbox_inches='tight')
