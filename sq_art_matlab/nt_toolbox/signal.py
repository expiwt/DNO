import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
from scipy import signal
from skimage import transform
from . import general as nt

def perform_convolution(x, h, bound="sym"):
    """
    perform_convolution - compute convolution with centered filter
    Works for 1D or 2D convolution
    """
    if bound not in ["sym", "per"]:
        raise Exception('bound should be sym or per')

    if np.ndim(x) == 3 and x.shape[2] < 4:
        # For color images
        y = x.copy()
        for i in range(x.shape[2]):
            y[:, :, i] = perform_convolution(x[:, :, i], h, bound)
        return y

    if np.ndim(x) == 3 and x.shape[2] >= 4:
        raise Exception('Not yet implemented for 3D array')

    n = x.shape
    p = np.array(h.shape)
    nd = np.ndim(x)

    if bound == 'sym':
        d1 = np.floor(p.astype(int) / 2).astype('int64')  # padding before
        d2 = p - d1 - 1  # padding after

        if nd == 1:
            # 1D convolution
            nx = len(x)
            xx = np.concatenate((x[d1[0]::-1], x, x[nx-1:nx-d2[0]-1:-1]))
            y = np.convolve(xx, h.flatten(), mode='valid')
            y = y[:nx]
            
        elif nd == 2:
            # 2D convolution
            nx, ny = x.shape
            xx = x.copy()
            
            # Vertical padding
            xx = np.vstack((xx[d1[0]-1::-1, :], xx, xx[nx-1:nx-d2[0]-1:-1, :]))
            # Horizontal padding  
            xx = np.hstack((xx[:, d1[1]-1::-1], xx, xx[:, ny-1:ny-d2[1]-1:-1]))
           
            y = signal.convolve2d(xx, h, mode='valid')
            y = y[d1[0]:d1[0]+n[0], d1[1]:d1[1]+n[1]]
            
    else:
        # Periodic boundary conditions
        if np.any(p > n):
            raise Exception('h filter should be shorter than x')
            
        n = np.array(n)
        p = np.array(p)
        d = np.floor((p - 1) / 2).astype(int)
        
        if nd == 1:
            h_padded = np.zeros(n)
            h_padded[d[0]:d[0]+p[0]] = h
            h_padded = np.roll(h_padded, -d[0])
            y = np.real(pyl.ifft(pyl.fft(x) * pyl.fft(h_padded)))
        else:
            h_padded = np.zeros(n)
            h_padded[d[0]:d[0]+p[0], d[1]:d[1]+p[1]] = h
            h_padded = np.roll(h_padded, -d[0], axis=0)
            h_padded = np.roll(h_padded, -d[1], axis=1)
            y = np.real(pyl.ifft2(pyl.fft2(x) * pyl.fft2(h_padded)))
            
    return y

def grad(f):
    """ Compute gradient of 2D image with periodic BC """
    s0 = np.concatenate((np.arange(1, f.shape[0]), [0]))
    s1 = np.concatenate((np.arange(1, f.shape[1]), [0]))
    g = np.dstack((f[s0, :] - f, f[:, s1] - f))
    return g

def div(g):
    """ Compute divergence of 2D vector field with periodic BC """
    s0 = np.concatenate(([g.shape[0]-1], np.arange(0, g.shape[0]-1)))
    s1 = np.concatenate(([g.shape[1]-1], np.arange(0, g.shape[1]-1)))
    f = (g[:, :, 0] - g[s0, :, 0]) + (g[:, :, 1] - g[:, s1, 1])
    return f