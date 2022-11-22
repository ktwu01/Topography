import numpy as np

def get_swath_data(orig_dem, orig_pr, orig_pet, orig_t, line_shape):
    # todo: clean up and make less hacky (especially nodata values)
    dist = orig_dem.distance
    dem_swath = np.array(orig_dem.dat)
    if (len(dist) == len(dem_swath) + 1):  # sometimes dist is longer than swath
        dist = orig_dem.distance[0:-1]
    dem_swath[dem_swath == -32768.] = np.nan  # Note: works only because this is returned as nodata value (WorldClim DEM)
    dem_swath[dem_swath == 32767.] = np.nan  # Note: works only because this is returned as nodata value (HydroSHEDS DEM)
    dem_swath[dem_swath == -9999.] = np.nan  # Note: works only because this is returned as nodata value (MERIT 250m DEM)
    isnan = np.isnan(dem_swath).any(axis=1)
    dem_swath = dem_swath[~isnan]
    # ma.masked_invalid(dem_swath)
    pr_swath = orig_pr.dat
    pr_swath = [d for (d, remove) in zip(pr_swath, isnan) if not remove]
    pr_swath = np.array(pr_swath)
    # pr_swath[pr_swath==-999] = np.nan
    # ma.masked_invalid(pr_swath)
    pet_swath = orig_pet.dat
    pet_swath = [d for (d, remove) in zip(pet_swath, isnan) if not remove]
    pet_swath = np.array(pet_swath)
    #pet_swath[pet_swath == 65535.] = np.nan  # Note: works only because this is returned as nodata value (CHELSA PET)
    #pet_swath[pet_swath == -32768.] = np.nan  # Note: works only because this is returned as nodata value (WorldClim PET)
    #isnan2 = np.isnan(pet_swath).any(axis=1)
    #pet_swath = pet_swath[~isnan2]
    # pet_swath[pet_swath==-999] = np.nan
    # pet_swath = pet_swath * 1000  # transform to Pa
    # ma.masked_invalid(pet_swath)
    t_swath = orig_t.dat
    t_swath = [d for (d, remove) in zip(t_swath, isnan) if not remove]
    t_swath = np.array(t_swath)
    # t_swath[t_swath==-999] = np.nan
    # ma.masked_invalid(t_swath)
    dist = dist[~isnan] + line_shape.xy[0][0] # works only if line goes from east to west

    return dist, dem_swath, pr_swath, pet_swath, t_swath
