import xarray as xr
import pandas as pd
from datetime import datetime as dt
from os.path import exists as file_exists
import os
from functools import reduce

# specify paths
data_path = r"C:/Users/Sebastian/Documents/Data/" #r"D:/Data/" #
results_path = "results/" #r"C:/Users/gnann/Documents/PYTHON/Topography/results/"
folder_path = "GlobSnow_v3.0_monthly_biascorrected_SWE/"
file_name = "_northern_hemisphere_monthly_biascorrected_swe_0.25grid.nc"

date_list = ["197902", "197903"]

for date in date_list:
    file_path = data_path + folder_path + date + file_name
    print(file_path)
    date = xr.open_dataset(file_path)
    d = date.resample(time="1Y").mean()
    d = d.mean("time")
