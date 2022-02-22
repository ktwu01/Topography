# helper function to define geometries

def get_region(region_name):
    # create geometries
    # line needs to be shorter than rectangle
    if region_name == 'Kilimanjaro':
        xy_line = [37.0, 37.8, -0.3, 0.1] # Kilimanjaro
        xy_box = [36.0, 39.0, -1.0, 1.0]
    elif region_name == 'Cascades':
        xy_line = [-124.5, -120.5, 45.0, 45.001] # Cascades
        xy_box = [-125.5, -119.5, 44.0, 46.0]
    elif region_name == 'NorthernAlps':
        xy_line = [11.0, 11.001, 47.7, 46.7] # Northern Alps
        xy_box = [10.0, 12.0, 46.0, 48.0]
    elif region_name == 'Himalaya':
        xy_line = [85.0, 87.0, 26.0, 29.0] # Himalaya
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