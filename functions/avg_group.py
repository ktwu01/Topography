import numpy as np

# This function calculates averages per latitude, i.e. it selects all cells with the same
# latitude and calculates the average (arithmetic mean). It optionally also uses a moving
# window to calculate smoother averages over multiple latitudinal values.

# todo: write function that works for all cases

def mean_group(vA0, vB0, nr_cells=1, thresh=1):
    vA, ind, counts = np.unique(vA0, return_index=True, return_counts=True) # get unique values in vA0
    vB = np.empty(len(vA))
    vB[:] = np.nan

    if nr_cells == 1:
    # calculate average per latitudinal cell
        thresh = 1
        for i in range(len(vA)):
            if len(vB0[np.where(vA0==vA[i])]) > thresh:
                vB[i] = np.nanmean(vB0[np.where(vA0==vA[i])])

    elif nr_cells == 3:
    # calculate average per latitudinal cell using a window of 3 cells, i.e. one before and one after
        for i in range(len(vA)):
            if i < 1:
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp3 = [vB0[np.where(vA0 == vA[i + 1])]]
                tmp = np.concatenate((tmp2, tmp3), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmean(tmp)
            elif i > (len(vA) - 2):
                tmp1 = [vB0[np.where(vA0 == vA[i - 1])]]
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp = np.concatenate((tmp1, tmp2), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmean(tmp)
            else:
                tmp1 = [vB0[np.where(vA0 == vA[i - 1])]]
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp3 = [vB0[np.where(vA0 == vA[i + 1])]]
                tmp = np.concatenate((tmp1, tmp2, tmp3), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmean(tmp)

    elif nr_cells == 5:
    # calculate average per latitudinal cell using a window of 5 cells, i.e. two before and two after
        for i in range(len(vA)):
            if i < 2:
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp3 = [vB0[np.where(vA0 == vA[i + 1])]]
                tmp4 = [vB0[np.where(vA0 == vA[i + 2])]]
                tmp = np.concatenate((tmp2, tmp3, tmp4), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmean(tmp)
            elif i > (len(vA) - 3):
                tmp0 = [vB0[np.where(vA0 == vA[i - 2])]]
                tmp1 = [vB0[np.where(vA0 == vA[i - 1])]]
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp = np.concatenate((tmp0, tmp1, tmp2), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmean(tmp)
            else:
                tmp0 = [vB0[np.where(vA0 == vA[i - 2])]]
                tmp1 = [vB0[np.where(vA0 == vA[i - 1])]]
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp3 = [vB0[np.where(vA0 == vA[i + 1])]]
                tmp4 = [vB0[np.where(vA0 == vA[i + 2])]]
                tmp = np.concatenate((tmp0, tmp1, tmp2, tmp3, tmp4), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmean(tmp)

    else:
        print('nr_cells parameter needs to be either 1, 3, or 5.')

    return vA, vB


# This function calculates median values per latitude, i.e. it selects all cells with
# the same latitude and calculates the median. It optionally also uses a moving window
# to calculate smoother median over multiple latitudinal values.

# todo: write function that works for all cases

def median_group(vA0, vB0, nr_cells=1, thresh=1):
    vA, ind, counts = np.unique(vA0, return_index=True, return_counts=True) # get unique values in vA0
    vB = np.empty(len(vA))
    vB[:] = np.nan

    if nr_cells == 1:
    # calculate average per latitudinal cell
        thresh = 1
        for i in range(len(vA)):
            if len(vB0[np.where(vA0==vA[i])]) > thresh:
                vB[i] = np.nanmedian(vB0[np.where(vA0==vA[i])])

    elif nr_cells == 3:
    # calculate average per latitudinal cell using a window of 3 cells, i.e. one before and one after
        for i in range(len(vA)):
            if i < 1:
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp3 = [vB0[np.where(vA0 == vA[i + 1])]]
                tmp = np.concatenate((tmp2, tmp3), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmedian(tmp)
            elif i > (len(vA) - 2):
                tmp1 = [vB0[np.where(vA0 == vA[i - 1])]]
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp = np.concatenate((tmp1, tmp2), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmedian(tmp)
            else:
                tmp1 = [vB0[np.where(vA0 == vA[i - 1])]]
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp3 = [vB0[np.where(vA0 == vA[i + 1])]]
                tmp = np.concatenate((tmp1, tmp2, tmp3), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmedian(tmp)

    elif nr_cells == 5:
    # calculate average per latitudinal cell using a window of 5 cells, i.e. two before and two after
        for i in range(len(vA)):
            if i < 2:
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp3 = [vB0[np.where(vA0 == vA[i + 1])]]
                tmp4 = [vB0[np.where(vA0 == vA[i + 2])]]
                tmp = np.concatenate((tmp2, tmp3, tmp4), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmedian(tmp)
            elif i > (len(vA) - 3):
                tmp0 = [vB0[np.where(vA0 == vA[i - 2])]]
                tmp1 = [vB0[np.where(vA0 == vA[i - 1])]]
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp = np.concatenate((tmp0, tmp1, tmp2), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmedian(tmp)
            else:
                tmp0 = [vB0[np.where(vA0 == vA[i - 2])]]
                tmp1 = [vB0[np.where(vA0 == vA[i - 1])]]
                tmp2 = [vB0[np.where(vA0 == vA[i])]]
                tmp3 = [vB0[np.where(vA0 == vA[i + 1])]]
                tmp4 = [vB0[np.where(vA0 == vA[i + 2])]]
                tmp = np.concatenate((tmp0, tmp1, tmp2, tmp3, tmp4), axis=None)
                if len(tmp) > thresh:
                    vB[i] = np.nanmedian(tmp)

    else:
        print('nr_cells parameter needs to be either 1, 3, or 5.')

    return vA, vB
