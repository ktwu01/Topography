import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import matplotlib.colors as ml_colors
import numpy as np
import pandas as pd
from cartopy import config
import cartopy.crs as ccrs
from brewer2mpl import brewer2mpl
from matplotlib.colors import LogNorm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import sys
import cmocean
import shapely.geometry as sgeom
from matplotlib.colors import LinearSegmentedColormap
from ast import literal_eval

o = brewer2mpl.get_map('YlGnBu', 'Sequential', 9, reverse=True)
c = o.mpl_colormap

# prepare data
data_path = r"C:\Users\Sebastian\Documents\Data\G3M\water_table_depth.csv"#r"C:\Users\gnann\Documents\Data\G3M\water_table_depth.csv"

df = pd.read_csv(data_path, sep=',')
var_name = 'WTD'

df_greenland = pd.read_csv("greenland.csv", sep=',')

plt.rcParams['axes.linewidth'] = 0.1

fig = plt.figure()
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()

bounds = np.linspace(0,50,11)
customnorm = ml_colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#norm=customnorm,
sc = ax.scatter(df['X'], df['Y'], norm=customnorm, transform=ccrs.PlateCarree(),
                marker='s',s=.35, edgecolors = 'none', c=df['WTD(m)'], cmap=c)

ax.scatter(df_greenland['lon'], df_greenland['lat'], transform=ccrs.PlateCarree(), norm=customnorm,
           marker='s', s=.35, edgecolors='none', c='lightgray')

ax.coastlines(linewidth=0.5)
#ax.set_clim(0, 1000)

box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
x0,y0,x1,y1 = box.bounds
ax.set_extent([x0,x1,y0,y1],ccrs.PlateCarree())

cbar = plt.colorbar(sc,orientation='horizontal', pad=0.01, shrink=.5)
cbar.set_label(var_name + ' [m]')
#cbar.set_ticks(np.linspace(0,100,11))
cbar.ax.tick_params(labelsize=6)
plt.gca().outline_patch.set_visible(False)

#plt.show()
plt.savefig("results/" + var_name + "_map.png", dpi=600,  bbox_inches='tight')
#fig.clear()
