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
data_path = "D:/Data/"

# check if folder exists
results_path = "./results/data_bias/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
df = pd.read_csv("data/data_bias.csv", sep=';')

#plt.bar(df["Landform"], 100*df["Global distribution"], 0.4, label="Global")
#plt.bar(df["Landform"], 100*df["Kratzert et al. (2023)"], 0.4, label="Kratzert")
#plt.bar(df["Landform"], 100*df["Moeck et al. (2020)"], 0.4, label="Moeck")
#plt.bar(df["Landform"], 100*df["Fan et al. (2013)"], 0.4, label="Fan")
#plt.show()

# todo: maybe plot dataset - global mean to highlight differences
N = 4
ind = np.arange(N)
width = 0.2
bar1 = plt.bar(ind, 100*df["Global distribution"], width, color='grey')
bar2 = plt.bar(ind + width, 100*df["Kratzert et al. (2023)"], width, color='tab:blue')
bar3 = plt.bar(ind + width * 2, 100*df["Moeck et al. (2020)"], width, color='tab:orange')
bar4 = plt.bar(ind + width * 3, 100*df["Fan et al. (2013)"], width, color='tab:brown')
#plt.xlabel("Landform")
plt.ylabel('Percentage %')
plt.xticks(ind + width*1.5, list(df["Landform"]))
plt.legend((bar1, bar2, bar3, bar4), ('Global', 'Kratzert', 'Moeck', 'Fan'))
plt.savefig(results_path + "data_bias"  + ".png", dpi=600, bbox_inches='tight')
plt.close()

# todo: maybe plot dataset - global mean to highlight differences
N = 4
ind = np.arange(N)
width = 0.2
plt.axhline(y=0.0, color='grey', linestyle='--')
bar2 = plt.bar(ind + width, 100*df["Kratzert et al. (2023)"]-100*df["Global distribution"], width, color='tab:blue')
bar3 = plt.bar(ind + width * 2, 100*df["Moeck et al. (2020)"]-100*df["Global distribution"], width, color='tab:orange')
bar4 = plt.bar(ind + width * 3, 100*df["Fan et al. (2013)"]-100*df["Global distribution"], width, color='tab:brown')
#plt.xlabel("Landform")
plt.ylabel('Percentage %')
plt.xticks(ind + width*1.5, list(df["Landform"]))
plt.ylim([-35,35])
plt.legend((bar2, bar3, bar4), ('Kratzert', 'Moeck', 'Fan'))
plt.savefig(results_path + "data_bias_rel"  + ".png", dpi=600, bbox_inches='tight')
plt.close()


x = ["Global", "Kratzert", "Moeck", "Fan"]
y0 = 100*df.loc[0].values[1:]
y1 = 100*df.loc[1].values[1:]
y2 = 100*df.loc[2].values[1:]+1 # to compensate for rounding error todo: recalculate more precisely
y3 = 100*df.loc[3].values[1:]+1
fig = plt.figure(figsize=(4, 2), constrained_layout=True)
width = 0.5
bar1 = plt.bar(x, y0, width=width, color='#ff7f00', label='Humid plains')
bar2 = plt.bar(x, y1, bottom=y0, width=width, color='#fdbf6f', label='Arid plains')
bar3 = plt.bar(x, y2, bottom=y0+y1, width=width, color='#1f78b4', label='Humid uplands')
bar4 = plt.bar(x, y3, bottom=y0+y1+y2, width=width, color='#a6cee3', label='Arid uplands')
#plt.xlabel("Landform")
plt.ylabel('Percentage %')
plt.ylim([0,100])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
#plt.legend()
plt.savefig(results_path + "data_bias_stacked"  + ".png", dpi=600, bbox_inches='tight')
plt.close()


x = ["Global", "Kratzert", "Moeck", "Fan"]
y0 = 100*df.loc[0].values[1:]
y1 = 100*df.loc[1].values[1:]
y2 = 100*df.loc[2].values[1:]+1 # to compensate for rounding error todo: recalculate more precisely
y3 = 100*df.loc[3].values[1:]+1
fig = plt.figure(figsize=(4, 2), constrained_layout=True)
width = 0.5
bar1 = plt.bar(x, y0, width=width, color='#01665e', label='Humid plains')
bar2 = plt.bar(x, y1, bottom=y0, width=width, color='#80cdc1', label='Arid plains')
bar3 = plt.bar(x, y2, bottom=y0+y1, width=width, color='#8c510a', label='Humid uplands')
bar4 = plt.bar(x, y3, bottom=y0+y1+y2, width=width, color='#dfc27d', label='Arid uplands')
#plt.xlabel("Landform")
plt.ylabel('Percentage %')
plt.ylim([0,100])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
#plt.legend()
plt.savefig(results_path + "data_bias_stacked"  + ".png", dpi=600, bbox_inches='tight')
plt.close()
