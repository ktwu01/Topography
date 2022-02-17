import fiona
import pandas as pd

def create_line(x1, x2, y1, y2):
# https://hatarilabs.com/ih-en/how-to-create-a-pointlinepolygon-shapefile-with-python-and-fiona-tutorial

    d = {'x': [x1, x2], 'y': [y1, y2], 'Name': ['Row1', 'Row1']}
    lineDf = pd.DataFrame(data=d)
    #lineDf = pd.read_csv('../Txt/cropLine.csv',header=0)
    #lineDf.head()

    schema = {
        'geometry':'LineString',
        'properties':[('Name','str')]
    }

    #open a fiona object
    lineShp = fiona.open('../results/tmp/cropLine.shp', mode='w', driver='ESRI Shapefile',
              schema = schema, crs = "EPSG:4326")

    #get list of points
    xyList = []
    rowName = ''
    for index, row in lineDf.iterrows():
        xyList.append((row.X,row.Y))
        rowName = row.Name

    #save record and close shapefile
    rowDict = {
    'geometry' : {'type':'LineString',
                     'coordinates': xyList},
    'properties': {'Name' : rowName},
    }
    lineShp.write(rowDict)
    #close fiona object
    lineShp.close()