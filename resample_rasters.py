from osgeo import gdal
import os

# This script resamples and aligns different rasters.

data_path = "/home/hydrosys/data/" #data_path = r"D:/Data/"
results_path = "/home/hydrosys/data/resampling/"

if not os.path.isdir(results_path):
    os.makedirs(results_path)

bounds = [-180, -90, 180, 90]

# 5 minute resolution
res = 5/60
res = 0.5/60

path_list = ["CHELSA/CHELSA_bio12_1981-2010_V.2.1.tif",
             "CHELSA/CHELSA_pet_penman_mean_1981-2010_V.2.1.tif",
             "CHELSA/CHELSA_bio1_1981-2010_V.2.1.tif",
             "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_12.tif",
             "WorldClim/7504448/global-et0_annual.tif/et0_yr/et0_yr.tif",
             "WorldClim/wc2.1_30s_bio/wc2.1_30s_bio_1.tif",
             "DEMs/MERIT_250m/dtm_elevation_merit.dem_m_250m_s0..0cm_2017_v1.0.tif",
             "Landforms/WorldLandform.tif"]

name_list = ["P_CHELSA",
             "PET_CHELSA",
             "T_CHELSA",
             "P_WorldClim",
             "PET_WorldClim",
             "T_WorldClim",
             "Elevation_MERIT",
             "WorldLandform"]

# if only a single file should be resampled
path_list = ["Global_Soil_Regolith_Sediment_1304/data/land_cover_mask.tif"]
name_list = ["Pelletier"]

for path, name in zip(path_list, name_list):
    print(name)
    ds_path = data_path + path
    ds = gdal.Open(ds_path)
    dsRes = gdal.Warp(results_path + name + "_30sec.tif", ds,
                      outputBounds=bounds, xRes=res, yRes=res, resampleAlg="med", dstSRS="EPSG:4326")
