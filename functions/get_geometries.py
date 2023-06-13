# helper function to define geometries

def get_swath_indices(region_name):
    # get indices of swaths to be plotted for different regions

    if region_name == 'Himalaya':
        ind = [10, 22]
        forcinglim = [0, 6000]

    elif region_name == 'Cascade Range':
        ind = [4, 9]
        forcinglim = [0, 3000]

    elif region_name == 'Pyrenees':
        ind = [1, 5]
        forcinglim = [0, 3000]

    elif region_name == 'European Alps':
        ind = [8, 14]
        forcinglim = [0, 3000]

    elif region_name == 'Cordillera Central Ecuador':
        ind = [3, 8]
        forcinglim = [0, 6000]

    elif region_name == 'Cordillera principal':
        ind = [3, 8]
        forcinglim = [0, 5000]

    elif region_name == 'Southern Andes':
        ind = [3, 11]
        forcinglim = [0, 4000]

    elif region_name == 'Sierra Madre del Sur':
        ind = [3, 12]
        forcinglim = [0, 3000]

    elif region_name == 'Sierra Nevada':
        ind = [6, 10]
        forcinglim = [0, 3000]

    elif region_name == 'Pegunungan Maoke':
        ind = [3, 10]
        forcinglim = [0, 5000]

    elif region_name == 'Albertine Rift Mountains':
        ind = [1, 5]
        forcinglim = [0, 3000]

    elif region_name == 'Ethiopian Highlands':
        ind = [15, 25]
        forcinglim = [0, 3000]

    else:
        raise('Region not defined.')

    return ind, forcinglim


def get_swath_geometries(region_name):
    # create geometries
    # line needs to be shorter than rectangle

    if region_name == 'Kilimanjaro':
        xy_line = [37.8, 36.9, -3.4, -2.7] # Kilimanjaro
        xy_box = [36.0, 39.0, -4.0, -2.0]

    elif region_name == 'Cascade Range':
        xy_line = [-124.0, -119.0, 45.0, 45.001] # Cascades
        xy_box = [-124.0, -119.0, 44.0, 46.0]

    elif region_name == 'Northern Alps':
        xy_line = [11.0, 11.001, 47.7, 46.7] # Northern Alps
        xy_box = [10.0, 12.0, 46.0, 48.0]

    elif region_name == 'European Alps':
        xy_line = [11.0, 11.001, 48.5, 45.5] # Alps
        xy_box = [9.0, 13.0, 44.0, 50.0]

    elif region_name == 'Sierra Nevada':
        xy_line = [-120.0, -118.001, 37.0, 38.5] # Sierra Nevada
        xy_box = [-121.0, -117.0, 36.0, 40.0]

    elif region_name == 'Cordillera Central Ecuador':
        xy_line = [-81.0, -76.0, -1.0, -1.0001] # Andes in Argentina and Chile
        xy_box = [-81.0, -76.0, -2.0, 0.0]

    elif region_name == 'Cordillera principal':
        xy_line = [-74.5, -69.5, -40.5, -40.5001] # Ecuador Andes
        xy_box = [-74.5, -69.5, -41.5, -39.5]

    elif region_name == 'Southern Andes':
        xy_line = [-74.0, -69.0, -39.0, -39.0001] # Ecuador Andes
        xy_box = [-74.0, -69.0, -40.0, -38.0]

    elif region_name == 'Himalaya':
        xy_line = [86.6, 86.8, 27.3, 28.1]
        xy_box = [84.0, 88.0, 25.0, 30.0]

    elif region_name == 'France':
        xy_line = [-1.0, 2.0, 45.0, 45.001] # France
        xy_box = [-2.0, 3.0, 44.0, 46.0]

    elif region_name == 'Ethiopian Highlands':
        xy_line = [36.0, 41.0, 7.0, 7.001] # Ethiopian Highlands
        xy_box = [36.0, 41.0, 6.0, 8.0]

    else:
        raise('Region not defined.')

    return xy_line, xy_box


def get_strike_geometries(region_name):
    # get data for different regions

    if region_name == 'Himalaya':
        line_path = "data/lines/Himalaya.shp"
        xlim = [75, 100]
        ylim = [24, 40]

    elif region_name == 'Sierra Madre del Sur':
        line_path = "data/lines/Sierra_Madre_Mexico.shp"
        xlim = [-105, -95]
        ylim = [15, 20]

    elif region_name == 'Cascade Range':
        line_path = "data/lines/Cascades.shp"
        xlim = [-125, -118]
        ylim = [40, 50]

    elif region_name == 'European Alps':
        line_path = "data/lines/European_Alps.shp"
        xlim = [6, 18]
        ylim = [42, 50]

    elif region_name == 'Cordillera Central Ecuador':
        line_path = "data/lines/Ecuadorian_Andes.shp"
        xlim = [-82, -76]
        ylim = [-8, 4]

    elif region_name == 'Cordillera principal':
        line_path = "data/lines/Chile_Andes.shp"
        xlim = [-75, -68]
        ylim = [-41, -33]

    elif region_name == 'Southern Andes':
        line_path = "data/lines/Southern_Andes.shp"
        xlim = [-75, -68]
        ylim = [-47, -33]

    elif region_name == 'Albertine Rift Mountains':
        line_path = "data/lines/Albertine_Rift_Mountains.shp"
        xlim = [26, 33]
        ylim = [-6, 2]

    elif region_name == 'Pegunungan Maoke':
        line_path = "data/lines/Pegunungan_Maoke.shp"
        xlim = [133, 143]
        ylim = [-6, -2]

    elif region_name == 'Pyrenees':
        line_path = "data/lines/Pyrenees.shp"
        xlim = [-4, 5]
        ylim = [41, 44]

    elif region_name == 'Sierra Nevada':
        line_path = "data/lines/Sierra_Nevada.shp"
        xlim = [-123, -117]
        ylim = [34, 43]

    elif region_name == 'Ethiopian Highlands':
        line_path = "data/lines/Ethiopian_Highlands.shp"
        xlim = [32, 43]
        ylim = [3, 20]

    else:
        raise('Region not defined.')

    return line_path, xlim, ylim
