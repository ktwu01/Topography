import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
import seaborn as sns
from functions import plotting_fcts
from functions import get_nearest_neighbour
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio

# This script ...

# prepare data
data_path = "data/CAMELS_table.csv"
data_path = "data/attributes/camels/attributes_hydroatlas_camels.csv"
data_path = "data/attributes/hysets/attributes_hydroatlas_hysets.csv"

# check if folder exists
results_path = "results/CAMELS/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
df = pd.read_csv(data_path, sep=',')

df["dummy"] = ""

#
x_name = "slp_dg_sav"
y_name = "gwt_cm_sav"
x_unit = " [m/km]"
y_unit = " [-]"
df["aridity_class"] = "energy-limited"
df.loc[df["ari_ix_sav"]/100 > 1, "aridity_class"] = "water-limited"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name, color="silver", marker='o', alpha=0.5, s=5, label=None)
#g.set(xlim=[1, 1000], ylim=[0, 1])
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
#g.map_dataframe(plotting_fcts.plot_means_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()
