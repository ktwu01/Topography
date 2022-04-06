def get_region(region_name):
    # get data for different regions

    if region_name == 'Himalaya':
        line_path = "lines/Himalaya_Arc.shp"
        xlim = [65, 105]
        ylim = [24, 40]

    elif region_name == 'Cascades':
        line_path = "lines/Cascades.shp"
        xlim = [-125, -118]
        ylim = [40, 50]

    elif region_name == 'European_Alps':
        line_path = "lines/European_Alps.shp"
        xlim = [4, 18]
        ylim = [42, 50]

    elif region_name == 'Ecuadorian_Andes':
        line_path = "lines/Ecuadorian_Andes.shp"
        xlim = [-82, -76]
        ylim = [-8, 4]

    else:
        raise('Region not defined.')

    return line_path, xlim, ylim
