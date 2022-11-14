import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import geopandas as gpd
from scipy import stats

#todo: clean up a bit...

# Creates plots of elevation vs. the climatic water balance for whole mountain ranges.

# specify paths
#data_path = r"C:/Users/Sebastian/Documents/Data/"
data_path = r"D:/Data/"
results_path = "results/"

shp_path = data_path + "GMBA mountain inventory V1.2(entire world)/GMBA Mountain Inventory_v1.2-World.shp"
#dem_path = data_path + "WorldClim/wc2.1_30s_elev/wc2.1_30s_elev.tif"
#pr_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_12.tif"
#pet_path = data_path + "WorldClim/7504448/global-et0_annual.tif/et0_yr/et0_yr.tif"
#t_path = data_path + "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_1.tif"
dem_path = data_path + "DEMs/hyd_glo_dem_30s/hyd_glo_dem_30s.tif"
pr_path = data_path + "CHELSA/CHELSA_bio12_1981-2010_V.2.1.tif"
pet_path = data_path + "CHELSA/CHELSA_pet_penman_mean_1981-2010_V.2.1.tif"
t_path = data_path + "CHELSA/CHELSA_bio1_1981-2010_V.2.1.tif"
p_name = "P"
pet_name = "PET"
t_name = "T"

# open raster and plot
dem = rxr.open_rasterio(dem_path, masked=True).squeeze() #todo: remove masked...
pr = rxr.open_rasterio(pr_path, masked=True).squeeze()
pet = rxr.open_rasterio(pet_path, masked=True).squeeze()
t = rxr.open_rasterio(t_path, masked=True).squeeze()


