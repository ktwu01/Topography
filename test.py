import json
import subprocess

data_path = r"D:/Data/" #r"C:/Users/Sebastian/Documents/Data/"
dataset_uri = data_path + r"Landforms/USGSEsriTNCWorldTerrestrialEcosystems2020/commondata/raster_data/WorldEcosystem.tif"
_rat = subprocess.check_output('gdalinfo -json ' + dataset_uri, shell=True)
data = json.loads(_rat) # load json string into dictionary
print(data)

# to get band-level data
bands = data['bands']
