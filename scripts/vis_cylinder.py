#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import griddata
from scipy.io import loadmat
import matplotlib.pylab as plt
from mayavi import mlab

from chebpy import cheb_barycentric_matrix

from scftpy import scft_contourf, SCFTConfig

def vis_cylinder(param='param.ini', data='scft_out', 
                 is_show=True, is_save=False):
    '''
    Assume the following PDE in cylindrical coordinates,
            du/dt = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/d^theta + d^2/dz^2) u -
            w*u
    with u = u(r,theta,z), w=w(r,theta,z) in the domain 
            [0, R] x [0, 2pi] x [0, Lz]
    for time t=0 to t=1,
    with boundary conditions
            d/dr[u(r=R,theta,z,t)] = ka u(r=R)
            u(r,theta,z,t) = u(r,theta+2pi,z,t) # periodic in theta direction
            u(r,theta,z,t) = u(r,theta,z+lz,t) # periodic in z direction
    '''
    if is_save:
        is_show = False
    config = SCFTConfig.from_file(param)
    Nt, Nz, Nr = config.grid.Lx, config.grid.Ly, config.grid.Lz
    R, Lz = config.uc.a, config.uc.b
    print Nt, Nz, Nr, R, Lz
    N2 = Nr /2
    Ntp = Nt + 1
    Nzp = Nz + 1
    Nrp = Nr

    mat = loadmat(data)
    phiA = mat['phiA']
    phiB = mat['phiB']
    
    # Periodic in x and z direction, Fourier
    ttp = np.linspace(0, 2*np.pi, Ntp)
    zzp = np.linspace(0, Lz, Nzp)
    # Non-periodic in r direction, Chebyshev
    ii = np.arange(Nr)
    rr = np.cos(np.pi * ii / (Nr-1)) # rr [-1, 1]
    rr = rr[:N2] # rr in (0, 1] with rr[0] = 1
    rrp = np.linspace(0, R, Nrp)
    zp, tp, rp = np.meshgrid(zzp, ttp, rrp)
    xp = rp * np.cos(tp)
    yp = rp * np.sin(tp)

    phiAp = np.zeros([Ntp, Nzp, N2])
    phiAp[:-1,:-1,:] = phiA
    phiAp[-1,:-1,:] = phiA[0,:,:]
    phiAp[:-1,-1,:] = phiA[:,0,:]
    phiAp[:-1,-1,:] = phiA[:,0,:]
    phiAp[-1,-1,:] = phiA[0,0,:]
    phiAp = cheb_interp3d_r(phiAp, rrp)
    phiBp = np.zeros([Ntp, Nzp, N2])
    phiBp[:-1,:-1,:] = phiB
    phiBp[-1,:-1,:] = phiB[0,:,:]
    phiBp[:-1,-1,:] = phiB[:,0,:]
    phiBp[-1,-1,:] = phiB[0,0,:]
    phiBp = cheb_interp3d_r(phiBp, rrp)

    xpp, ypp, zpp = np.mgrid[-R:R:Nrp*1j,
                             -R:R:Nrp*1j,
                             0:Lz:Nzp*1j]
    phiApp = griddata((xp.ravel(),yp.ravel(),zp.ravel()), phiAp.ravel(), 
                   (xpp, ypp, zpp), method='linear')
    phiBpp = griddata((xp.ravel(),yp.ravel(),zp.ravel()), phiBp.ravel(), 
                   (xpp, ypp, zpp), method='linear')
    phiABpp = phiApp - phiBpp
    if is_show:
        mlab.contour3d(xpp, ypp, zpp, phiABpp, 
                       contours=64, transparent=True, colormap='Spectral')
        mlab.show()
    if is_save:
        mlab.contour3d(xpp, ypp, zpp, phiABpp, 
                       contours=64, transparent=True, colormap='Spectral')
        mlab.savefig('phiAB.png')


def cheb_interp3d_r(u, vr):
    '''
    Use chebyshev interpolation for the last dimension of cylindrical
    coordinates (theta, z, r).
    u(theta, z, r): source data, note that the range of r is (0, 1]
    vr: vector to be interpolated, size is Nrp.
    '''
    Nt, Nz, N2 = u.shape
    Nrp = vr.size
    uout = np.zeros([Nt, Nz, Nrp])
    vrp = np.linspace(0, 1, Nrp)
    T = cheb_barycentric_matrix(vrp, 2*N2-1)
    #print Nt, Nz, Nr, Nrp, T.shape, u[0,0].shape
    for i in xrange(Nt):
        for j in xrange(Nz):
            up = u[i,j]
            up = np.hstack((up, up[::-1]))
            uout[i,j] = np.dot(T, up)
    return uout


if __name__ == '__main__':
    vis_cylinder()

