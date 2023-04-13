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
results_path = "./results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load and process data
df = pd.read_csv("data/data_bias.csv", sep=';')

plt.bar(df["Landform"], 100*df["Global distribution"], 0.4, label="Global")
plt.bar(df["Landform"], 100*df["Kratzert et al. (2023)"], 0.4, label="Kratzert")
plt.bar(df["Landform"], 100*df["Moeck et al. (2020)"], 0.4, label="Moeck")
plt.bar(df["Landform"], 100*df["Fan et al. (2013)"], 0.4, label="Fan")
plt.show()

# todo: maybe plot dataset - global mean to highlight differences
N = 4
ind = np.arange(N)
width = 0.2
bar1 = plt.bar(ind, 100*df["Global distribution"], width, color='grey')
bar2 = plt.bar(ind + width, 100*df["Kratzert et al. (2023)"], width, color='tab:blue')
bar3 = plt.bar(ind + width * 2, 100*df["Moeck et al. (2020)"], width, color='tab:orange')
bar4 = plt.bar(ind + width * 3, 100*df["Fan et al. (2013)"], width, color='tab:red')
#plt.xlabel("Landform")
plt.ylabel('Percentage %')
plt.xticks(ind + width*1.5, list(df["Landform"]))
plt.legend((bar1, bar2, bar3, bar4), ('Global', 'Kratzert', 'Moeck', 'Fan'))
plt.show()
