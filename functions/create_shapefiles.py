from shapely.geometry import mapping
from shapely import geometry
import fiona

def create_line_shp(xy_line, path="/shapefiles/line.shp"):

    line = geometry.LineString([geometry.Point(xy_line[0], xy_line[2]),
                                geometry.Point(xy_line[1], xy_line[3])])

    schema = {'geometry': 'LineString', 'properties': {'id': 'int'}}
    # write a new shapefile
    with fiona.open(path, 'w', 'ESRI Shapefile', schema) as c:
        c.write({'geometry': mapping(line), 'properties': {'id': 123}})

def create_polygon_shp(polygon, path="/shapefiles/polygon.shp"):

    schema = {'geometry': 'Polygon', 'properties': {'id': 'int'}}
    # write a new shapefile
    with fiona.open(path, 'w', 'ESRI Shapefile', schema) as c:
        c.write({'geometry': mapping(polygon), 'properties': {'id': 123}})
