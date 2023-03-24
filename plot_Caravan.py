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
from pingouin import partial_corr

# This script ...

# prepare
data_path = "D:/Data/"
caravan_path = "data/complete_table.csv" #"D:/Data/Caravan/complete_table.csv" #
#data_path = "data/attributes/camels/attributes_hydroatlas_camels.csv"
#data_path = "data/attributes/hysets/attributes_hydroatlas_hysets.csv"

slope_path = data_path + "resampling/" + "Slope_MERIT_30s.tif"
#elevation_path = data_path + "resampling/" + "Elevation_MERIT_30s.tif"
landform_path = data_path + "resampling/" + "WorldLandform_30sec.tif"

slope = rio.open(slope_path, masked=True)
#elevation = rio.open(elevation_path, masked=True)
landform = rio.open(landform_path, masked=True)

# check if folder exists
results_path = "results/Caravan/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
df = pd.read_csv(caravan_path, sep=',')

coord_list = [(x, y) for x, y in zip(df['gauge_lon'], df['gauge_lat'])]

df['slope_30s'] = [x for x in slope.sample(coord_list)]
df['slope_30s'] = np.concatenate(df['slope_30s'].to_numpy())
df.loc[df["slope_30s"] < 0, "slope_30s"] = np.nan
df['slope_30s'] = np.tan(np.deg2rad(df['slope_30s'] * 0.01))

df['landform'] = [x for x in landform.sample(coord_list)]
df['landform'] = np.concatenate(df['landform'].to_numpy())
df.loc[df["landform"] < 1, "landform"] = np.nan
df.loc[df["landform"] > 4, "landform"] = np.nan

df["dummy"] = ""

# BFI per landform
print("BFI per landform")
print("Plains " + ": " + str(round(df.loc[df["landform"]==4, "BFI"].mean(), 2)))
print("Tablelands " + ": " + str(round(df.loc[df["landform"]==3, "BFI"].mean(), 2)))
print("Hills " + ": " + str(round(df.loc[df["landform"]==2, "BFI"].mean(), 2)))
print("Mountains " + ": " + str(round(df.loc[df["landform"]==1, "BFI"].mean(), 2)))
#print("Uplands " + ": " + str(round(df.loc[np.logical_or(df["landform"]==1, df["landform"]==2, df["landform"]==3), "BFI"].mean(), 2)))
print("Uplands " + ": " + str(round(df.loc[df["landform"]<4, "BFI"].mean(), 2)))

df["slope"] = df["slp_dg_sav"] * 0.1#np.tan(np.deg2rad(df['slp_dg_sav'] * 0.1)) #
df["aridity_class"] = "energy-limited"
df.loc[100/df["ari_ix_sav"] > 1, "aridity_class"] = "water-limited"

# slope
x_name = "slope"
y_name = "BFI"
x_unit = " [deg]"
y_unit = " [-]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name, color="silver", marker='o', lw=0, alpha=1, s=5, label=None)
g.set(xlim=[0.1, 100], ylim=[0, 1])
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

print(x_name + " and " + y_name)
r, p = stats.spearmanr(df[x_name], df[y_name])
print(str(np.round(r,2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="frac_snow", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="ari_ix_sav", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))

# landform
x_name = "slope_30s"
y_name = "BFI"
z_name = "landform"
x_unit = " [deg]"
y_unit = " [-]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
df["hue"] = np.round(df[z_name],0) # to have fewer unique values
g = sns.FacetGrid(df, col="dummy", hue="hue", palette="viridis", col_wrap=4)
g.map_dataframe(sns.scatterplot, x_name, y_name, marker='o', lw=0, alpha=0.75, s=3, label=None)
#plt.scatter(df[x_name], df[y_name], marker='o', lw=0, alpha=1, s=5, label=None)
g.set(xlim=[0.001, 1], ylim=[0, 1])
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
#g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_landform.png", dpi=600, bbox_inches='tight')
plt.close()

print(x_name + " and " + y_name)
r, p = stats.spearmanr(df[x_name], df[y_name])
print(str(np.round(r,2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="frac_snow", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="ari_ix_sav", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))

# slope
x_name = "slope"
y_name = "BFI"
z_name = "frac_snow"
x_unit = " [deg]"
y_unit = " [-]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
df["hue"] = np.round(df[z_name],2) # to have fewer unique values
g = sns.FacetGrid(df, col="dummy", hue="hue", palette="viridis", col_wrap=4)
g.map_dataframe(sns.scatterplot, x_name, y_name, marker='o', lw=0, alpha=1, s=5, label=None)
#plt.scatter(df[x_name], df[y_name], marker='o', lw=0, alpha=1, s=5, label=None)
g.set(xlim=[0.001, 1], ylim=[0, 1])
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
#g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_frac_snow.png", dpi=600, bbox_inches='tight')
plt.close()

print(x_name + " and " + y_name)
r, p = stats.spearmanr(df[x_name], df[y_name])
print(str(np.round(r,2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="frac_snow", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="ari_ix_sav", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))

# stream gradient
x_name = "sgr_dk_sav"
y_name = "BFI"
x_unit = " [deg]"
y_unit = " [-]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name, color="silver", marker='o', lw=0, alpha=1, s=5, label=None)
#g.set(xlim=[1/1000, 1000/1000], ylim=[0, 1])
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()


# test for partial correlation
fig = plt.figure(figsize=(3, 2))
ax = plt.axes()
plt.grid(color='grey', linestyle='--', linewidth=0.25)
count, bins_count = np.histogram(df["slope"], bins=1000)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], cdf, color="grey", label="Global distribution")
ax.set_xlabel("Slope [m/m]")
ax.set_ylabel("Cumulative probability [-]")
ax.set_xlim([0, 0.5])
plt.savefig(results_path + "slope_histogram.png", dpi=600, bbox_inches='tight')
plt.close()


# total runoff ratio
x_name = "slope"
y_name = "TotalRR"
x_unit = " [deg]"
y_unit = " [-]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name, color="silver", marker='o', lw=0, alpha=1, s=5, label=None)
g.set(xlim=[0.1, 100], ylim=[0, 1])
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

print(x_name + " and " + y_name)
r, p = stats.spearmanr(df[x_name], df[y_name], nan_policy='omit')
print(str(np.round(r,2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="frac_snow", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="ari_ix_sav", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))

# event runoff ratio
x_name = "slope"
y_name = "EventRR"
x_unit = " [deg]"
y_unit = " [-]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name, color="silver", marker='o', lw=0, alpha=1, s=5, label=None)
g.set(xlim=[0.1, 100], ylim=[0, 1])
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

print(x_name + " and " + y_name)
r, p = stats.spearmanr(df[x_name], df[y_name], nan_policy='omit')
print(str(np.round(r,2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="frac_snow", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="ari_ix_sav", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))

# xxx
x_name = "slope"
y_name = "EventGraphThresholds_7"
x_unit = " [deg]"
y_unit = " [-]"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name, color="silver", marker='o', lw=0, alpha=1, s=5, label=None)
g.set(xlim=[0.1, 100], ylim=[0, 1])
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="aridity_class", group="energy-limited")
#g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:orange", group_type="aridity_class", group="water-limited")
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:red", group_type="dummy", group="")
g.add_legend(loc=(.2, .75), handletextpad=0.0)
# results_df = plotting_fcts.binned_stats_table(df, x_name, y_name, sources)
g.set(xlabel = x_name + x_unit, ylabel = y_name + y_unit)
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='linear')
plt.savefig(results_path + x_name + '_' + y_name + "_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

print(x_name + " and " + y_name)
r, p = stats.spearmanr(df[x_name], df[y_name], nan_policy='omit')
print(str(np.round(r,2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="frac_snow", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))
r_partial_mat = partial_corr(data=df, x=x_name, y=y_name, covar="ari_ix_sav", method='spearman')
print(str(np.round(r_partial_mat.r.values[0],2)))
