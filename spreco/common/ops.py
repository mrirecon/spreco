from scipy.fftpack import fftshift, ifftshift, ifft2, fft2
import numpy as np

def mfft2(x, dims, axis=(0,1), center=False):
    nx, ny = dims
    if center:
        return ifftshift(fft2(fftshift(x), axes=axis))/np.sqrt(nx*ny)
    else:
        return fftshift(fft2(fftshift(x), axes=axis))/np.sqrt(nx*ny)

def mifft2(x, dims, axis=(0,1), center=False):
    nx, ny = dims
    if center:
        return fftshift(ifft2(ifftshift(x), axes=axis))*np.sqrt(nx*ny)
    else:
        return fftshift(ifft2(fftshift(x), axes=axis))*np.sqrt(nx*ny)


def A_cart(img, coilsen, mask, shape, axis=(0,1), center=False):
    """
    forward cartesian A 
    """
    coil_img = coilsen*img[..., np.newaxis]
    kspace = mfft2(coil_img, shape, axis=axis, center=center)
    kspace = np.multiply(kspace, mask[...,np.newaxis])
    return kspace

def AT_cart(kspace, coilsen, mask, shape, axis=(0,1), center=False):
    """
    adjoint cartesian AT
    coil dimension should always be the last
    """
    coil_img = mifft2(kspace*mask[...,np.newaxis], shape, axis=axis, center=center)
    coil_sum = np.sum(coil_img*np.conj(coilsen), axis=-1)
    return coil_sum

def sense_kernel_cart(AHy, img_k, coilsen, mask, shape, step_size=1, axis=(1,2), center=False):
    tmp = A_cart(img_k, coilsen, mask, shape, axis=axis, center=center)
    tmp = AT_cart(tmp, coilsen, mask, shape, axis=axis, center=center)
    img_k = img_k + step_size*(AHy - tmp)
    return img_k

def AHA(x, coilsen, mask, shape, axis=(1,2), center=False):
    tmp = A_cart(x, coilsen, mask, shape, axis, center)
    ret = AT_cart(tmp, coilsen, mask, shape, axis, center)
    return ret