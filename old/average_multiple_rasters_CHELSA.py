import rioxarray as rxr

# specify paths
#data_path = r"C:/Users/Sebastian/Documents/Data/"
data_path = r"D:/Data/"
results_path = "../results/"

#raster_path = data_path + "/wc2.1_30s_vapr/wc2.1_30s_vapr_"
raster_path = data_path + "CHELSA/pet/" + "CHELSA_pet_penman_"
file_ending = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# loop over all rasters
raster = rxr.open_rasterio(raster_path + file_ending[0] + "_1981-2010_V.2.1.tif", masked=True).squeeze()
for i in range(1,12):
    tmp = rxr.open_rasterio(raster_path + file_ending[i] + "_1981-2010_V.2.1.tif", masked=True).squeeze()
    raster.values = raster.values + tmp.values
raster.values = raster.values#/12

# save raster
raster.rio.to_raster(results_path + "CHELSA_pet_penman_year_1981-2010_V.2.1.tif")

#pet_test = rxr.open_rasterio("D:/Data/CHELSA/CHELSA_pet_penman_mean_1981-2010_V.2.1.tif", masked=True).squeeze()
