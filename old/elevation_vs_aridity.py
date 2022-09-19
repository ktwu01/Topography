import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from functions import plotting_fcts
from scipy import stats

# specify paths
data_path = "/home/hydrosys/data/" #r"C:/Users/Sebastian/Documents/Data/"
data_path = r"D:/Data/"
results_path = "../results/"

dem_path = data_path + "WorldClim/" + "wc2.1_30s_elev/wc2.1_30s_elev.tif"
slope_path = data_path + "Geomorpho90m/" + "dtm_slope_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
conv_path = data_path + "Geomorpho90m/" + "dtm_convergence_merit.dem_m_250m_s0..0cm_2018_v1.0.tif"
pr_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_12.tif"
pet_path = data_path + "WorldClim/7504448/global-et0_annual.tif/et0_yr/et0_yr.tif"
ai_path = data_path + "WorldClim/7504448/global-ai_et0/ai_et0/ai_et0.tif"
t_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_1.tif"

# open raster
#dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
dem = rio.open(dem_path, masked=True)
slope = rio.open(slope_path, masked=True)
conv = rio.open(conv_path, masked=True)
pr = rio.open(pr_path, masked=False)
pet = rio.open(pet_path, masked=False)
ai = rio.open(ai_path, masked=True)
t = rio.open(t_path, masked=True)

#x = ai.read().squeeze() # only works for same grids
y = dem.read().squeeze().flatten()
y = y.astype(np.float32)
y[y<-30000] = np.nan

from rasterio.merge import merge
x, _ = merge([ai, dem])
x = x.squeeze().flatten()
x = x.astype(np.float32)
x[x<-30000] = np.nan
x = x/10000

isn = (np.isfinite(x) & np.isfinite(y))
x = x[isn]
y = y[isn]

df = pd.DataFrame()
df["x"] = x
df["y"] = y

#n=300000000
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()#projection=ccrs.Robinson()
#ax.scatter(df["x"].sample(n), df["y"].sample(n), s=5, facecolor='tab:grey', edgecolor='none', alpha=0.1)
plotting_fcts.plot_bins(df["y"], df["x"])
ax.set_ylabel('P/PET [-]')
ax.set_xlabel('Elevation [m]')
ax.set_ylim([0.05, 5])
ax.set_yscale('log')
ax.set_xlim([10, 10000])
ax.set_xscale('log')
rho_s, _ = stats.spearmanr(df["y"],df["x"])
ax.annotate("rho_s: {:.2f} ".format(rho_s), xy=(.1, .9), xycoords=ax.transAxes, fontsize=10)
plt.savefig(results_path + "elev_vs_aridity.png", dpi=600, bbox_inches='tight')
plt.close()
