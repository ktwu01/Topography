# helper function to define geometries

def get_swath_geometries(region_name):
    # create geometries
    # line needs to be shorter than rectangle
    if region_name == 'Kilimanjaro':
        xy_line = [37.8, 36.9, -3.4, -2.7] # Kilimanjaro
        xy_box = [36.0, 39.0, -4.0, -2.0]
        #xy_line = [37.0, 37.8, -0.3, 0.1] # Kilimanjaro
        #xy_box = [36.0, 39.0, -1.0, 1.0]
    elif region_name == 'Cascades':
        xy_line = [-124.5, -120.5, 45.0, 45.001] # Cascades
        #xy_box = [-125.5, -119.5, 44.0, 46.0]
        xy_box = [-125.5, -119.5, 44.0, 46.0]
    elif region_name == 'Northern_Alps':
        xy_line = [11.0, 11.001, 47.7, 46.7] # Northern Alps
        xy_box = [10.0, 12.0, 46.0, 48.0]
    elif region_name == 'European_Alps':
        xy_line = [11.0, 11.001, 48.5, 45.5] # Alps
        xy_box = [9.0, 13.0, 44.0, 50.0]
    elif region_name == 'Sierra_Nevada':
        xy_line = [-120.0, -118.001, 37.0, 38.5] # Sierra Nevada
        xy_box = [-121.0, -117.0, 36.0, 40.0]
    elif region_name == 'Ecuadorian_Andes':
        xy_line = [-80.5, -76.5, -1.0, -1.0001] # Ecuador Andes
        xy_box = [-82.0, -76.0, -4.0, 1.0]
    elif region_name == 'Himalaya':
        #xy_line = [85.0, 87.0, 26.0, 29.0] # Himalaya
        xy_line = [86.6, 86.8, 27.3, 28.1]
        xy_box = [84.0, 88.0, 25.0, 30.0]
    elif region_name == 'France':
        xy_line = [-1.0, 2.0, 45.0, 45.001] # France
        xy_box = [-2.0, 3.0, 44.0, 46.0]
    elif region_name == 'xxx':
        xy_line = [0.0, 0.0, 0.0, 0.0] # xxx
        xy_box = [0.0, 0.0, 0.0, 0.0]
    else:
        raise('Region not defined.')

    return xy_line, xy_box


def get_strike_geometries(region_name):
    # get data for different regions

    if region_name == 'Himalaya':
        line_path = "data/lines/Himalaya_Arc.shp"
        xlim = [65, 105]
        ylim = [24, 40]

    elif region_name == 'Cascade Range':
        line_path = "data/lines/Cascades.shp"
        xlim = [-125, -118]
        ylim = [40, 50]

    elif region_name == 'European Alps':
        line_path = "data/lines/European_Alps.shp"
        xlim = [4, 18]
        ylim = [42, 50]

    elif region_name == 'Cordillera Central Ecuador':
        line_path = "data/lines/Ecuadorian_Andes.shp"
        xlim = [-82, -76]
        ylim = [-8, 4]

    else:
        raise('Region not defined.')

    return line_path, xlim, ylim
