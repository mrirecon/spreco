import numpy as np

def gen_mask_1D(ratio=0.1,center=20, ph=256, fe=256):
    """
    generate undersampling mask along 1 dimension
    Args:

    ratio: sampling ratio
    center: center lines retained
    fe: frequency encoding
    ph: phase encoding lines

    Returns:
    mask
    """
    k = int(round(ph*ratio)/2.0)
    ma = np.zeros(ph)
    ri = np.random.choice(int(ph/2-center/2), k, replace=False)
    ma[ri] = 1
    ri = np.random.choice(int(ph/2-center/2), k, replace=False)
    ma[ri+int(ph/2+center/2)] = 1
    ma[int(ph/2-center/2): int(ph/2+center/2)] = 1
    mask = np.tile(ma, [fe, 1])
    return mask

def gen_mask_2D( nx, ny, center_r = 20, undersampling = 0.3 ):
    #create undersampling mask
    k = int(round(nx*ny*undersampling)) #undersampling
    ri = np.random.choice(nx*ny,k,replace=False) #index for undersampling
    ma = np.zeros(nx*ny) #initialize an all zero vector
    ma[ri] = 1 #set sampled data points to 1
    mask = ma.reshape((nx,ny))

    # center k-space index range
    if center_r > 0:

        cx = np.int(nx/2)
        cy = np.int(ny/2)

        cxr_b = round(cx-center_r)
        cxr_e = round(cx+center_r+1)
        cyr_b = round(cy-center_r)
        cyr_e = round(cy+center_r+1)

        mask[cxr_b:cxr_e, cyr_b:cyr_e] = 1. #center k-space is fully sampled

    return mask