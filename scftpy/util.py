# -*- coding: utf-8 -*-
"""
util
====

Utilities for SCFT calculations.

"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from chebpy import cheb_barycentric_matrix


__all__ = ['quad_open4',
           'quad_semiopen4',
           'quad_semiopen3',
           'scft_contourf',
           'contourf_slab2d',
           'list_datafile',
           ]


def quad_open4(f, dx):
    '''
    Integrate f[0..N] with open interval, (0, N).
    int_f = dx * (55/24*f_1 - 1/6*f_2 + 11/8 *f_3 + f_4 + f_5 + f_6
                  ... + f_{N-4} + 11/8*f_{N-3}- 1/6*f_{N-2} + 55/24*f_{N-1})
    '''
    N = f.size - 1
    q = 55./24 * f[1] - 1./6 * f[2] + 11./8 * f[3]
    q += 55./24 * f[N-1] - 1./6 * f[N-2] + 11./8 * f[N-3]
    for i in xrange(4, N-3):
        q += f[i]

    return q * dx


def quad_semiopen4(f, dx):
    '''
    Integrate f[0..N] with semi-open interval, (0, N].
    int_f = dx * (55/24*f_1 - 1/6*f_2 + 11/8 *f_3 + f_4 + f_5 + f_6
                  ... + f_{N-3} + 23/24*f_{N-3}- 7/6*f_{N-2} + 3/8*f_{N-1})
    '''
    N = f.size - 1
    q = 55./24 * f[1] - 1./6 * f[2] + 11./8 * f[3]
    q += 3./8 * f[N] - 7./6 * f[N-1] + 23./24 * f[N-2]
    for i in xrange(4, N-2):
        q += f[i]

    return q * dx


def quad_semiopen3(f, dx):
    '''
    Integrate f[0..N] in semi-open interval, (0, N].
    int_f = dx * (55/24*f_1 - 1/6*f_2 + 11/8 *f_3 + f_4 + f_5 + f_6
                  ... + f_{N-4} + 11/8*f_{N-3}- 1/6*f_{N-2} + 55/24*f_{N-1})
    '''
    N = f.size - 1
    q = 23./12 * f[1] + 7./12 * f[2]
    q += 5./12 * f[N] + 13./12 * f[N-1]
    for i in xrange(3, N-2):
        q += f[i]

    return q * dx


def scft_contourf(x, y, z, levels=None, cmap=None, show_cbar=False, **kwargs):
    dx = x.max() - x.min()
    dy = y.max() - y.min()
    w, h = plt.figaspect(float(dy/dx))  # float is must
    # No frame, white background, w/h aspect ratio figure
    fig = plt.figure(figsize=(w/2, h/2), frameon=False, dpi=150, facecolor='w')
    # full figure subplot, no boarder, no axes
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, axisbg='w')
    # no ticks
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Default: there are 256 contour levels
    if levels is None:
        step = (z.max() - z.min()) / 32
        levels = np.arange(z.min(), z.max()+step, step)
    # Default: colormap is Spectral from matplotlib.cm
    if cmap is None:
        cmap = plt.cm.Spectral
    # actual plot
    cf = ax.contourf(x, y, z, levels=levels, cmap=cmap,
                     antialiased=False, **kwargs)
    if show_cbar:
        plt.colorbar(cf)
    return fig


def cheb_interp2d_y(u, vy):
    '''
    Use chebyshev interpolation for the last dimension of Cartesian coordinates
    (x, y).
    u(x, y): source data
    vy: vector to be interpolated, size is Nyp.
    '''
    Nx, Ny = u.shape
    Nyp = vy.size
    uout = np.zeros([Nx, Nyp])
    vyp = np.linspace(-1, 1, Nyp)
    T = cheb_barycentric_matrix(vyp, Ny-1)
    for i in xrange(Nx):
        uout[i] = np.dot(T, u[i])
    return uout


def contourf_slab2d(data, Lx, Ly):
    '''
    data is a 2D array with shape (Nx, Ny) to be interpolated to an array with shape of (Nx+1, 2*Ny), the corresponding physical dimension
    is (Lx, Ly).
    Regular grid along x, and Chebyshev grid along y.
    '''
    Nx, Ny = data.shape
    Nxp = Nx + 1
    Nyp = 4 * Ny

    # Periodic in x direction, regular grid
    xxp = np.linspace(0, Lx, Nxp)
    # Non-periodic in y direction, Chebyshev grid
    #ii = np.arange(Ny)
    #yy = np.cos(np.pi * ii / (Ny - 1))  # rr [-1, 1]
    yyp = np.linspace(0, Ly, Nyp)
    yp, xp = np.meshgrid(yyp, xxp)

    datap = np.zeros((Nxp, Ny))
    datap[:-1, :] = data
    datap[-1, :] = data[0, :]
    datap = cheb_interp2d_y(datap, yyp)
    scft_contourf(xp, yp, datap)

    return xxp, yyp, datap


def list_datafile(path='.', prefix='scft_out'):
    '''
    path: the path where datafile located.
    prefix: the prefix of datafile name.
    '''
    datafiles = []
    for f in os.listdir(path):
        p = os.path.join(path, f)  # path
        if os.path.isdir(p):
            pt = os.path.join(p, prefix+'_*.mat')  # path to be globbed
            files = glob.glob(pt)
            fnames = [os.path.basename(x) for x in files]
            data_name = get_final_datafile(fnames)
            if data_name == '':
                print p, ' data file missing.'
                continue
            dfile = os.path.join(p, data_name)
            datafiles.append(dfile)

    return datafiles


def get_final_datafile(namelist):
    '''
    Each name has the form 'scft_out_XXXX.mat', where XXXX is a number.
    '''
    datafile = ''
    num = 0  # a number to be compared
    for f in namelist:
        name, ext = os.path.splitext(f)  # split into 'scft_out_XXXX', '.mat'
        fragments = name.split('_')  # split into 'scft', 'out', 'XXXX'
        n = int(fragments[-1])
        if n > num:
            num = n
            datafile = name
    return datafile
