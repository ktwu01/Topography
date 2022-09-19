import rioxarray as rxr

# specify paths
data_path = r"C:/Users/Sebastian/Documents/Data/"
#data_path = r"D:/Data/"
results_path = "../results/"

raster_path = data_path + "/wc2.1_30s_vapr/wc2.1_30s_vapr_"
file_ending = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# loop over all rasters
raster = rxr.open_rasterio(raster_path + file_ending[0] + ".tif").squeeze()#, masked=True).squeeze()
for i in range(1,12):
    tmp = rxr.open_rasterio(raster_path + file_ending[i] + ".tif").squeeze()#, masked=True).squeeze()
    raster.values = raster.values + tmp.values
raster.values = raster.values/12

# save raster
raster.rio.to_raster(results_path + "wc2.1_30s_vapr_avg.tif")
