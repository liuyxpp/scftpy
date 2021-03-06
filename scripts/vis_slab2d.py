#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
vslab2d
=======

A script for displaying and saving density distribution generated by
scft_confined/SlabXX2d, where XX stands for the polymer model. For A-B diblock
copolymer, XX=AB.

Copyright (C) 2013 Yi-Xin Liu (lyx@fudan.edu.cn)

'''

import argparse
import os
import glob
import json
from ConfigParser import SafeConfigParser

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pylab as plt

from chebpy import cheb_barycentric_matrix

from scftpy import scft_contourf, SCFTConfig

parser = argparse.ArgumentParser(description='vis_slab2d options')

parser.add_argument('-b', '--batch',
                    action='store_true',
                    help='If present or True, perform batch mode.')
parser.add_argument('-r', '--path',
                    default='.',
                    help='Path to be processed, for batch mode use.')
parser.add_argument('-p', '--param_file',
                    default='param.ini',
                    help='SCFT configuration file, *.ini')
parser.add_argument('-d', '--data_file',
                    default='scft_out',
                    help='SCFT generated data file, *.mat')
parser.add_argument('-s', '--save',
                    action='store_true',
                    help='If present or True, save figure.')
args = parser.parse_args()


def batch_vis_slab2d(path='.', param='param.ini', data='scft_out'):
    '''
    Batch mode vis_slab2d, for all directory in the <path>.
    Note: ONLY directories in the path will be processed.
    The input var <data> is the main part of the whole data file name. The
    suffix is in the form '_XXXX', where XXXX is the max number of time steps.
    Then the full data file name is 'scft_out_XXXX.mat', the '.mat' can be
    ignored.
    Generated figures are stored in the same directory as its data file.
    Other data, such as H, are stored in the parent path as 'data.mat'.
    '''
    is_save = True  # Do not show figure in the batch mode

    var = []
    F = []

    for f in os.listdir(path):
        p = os.path.join(path, f)  # path
        if os.path.isdir(p):
            pt = os.path.join(p, data+'_*.mat')  # path to be globbed
            datafiles = glob.glob(pt)
            fnames = [os.path.basename(x) for x in datafiles]
            data_name = get_final_datafile(fnames)
            if data_name == '':
                print p, ' data file missing.'
                continue
            pfile = os.path.join(p, param)
            if not os.path.exists(pfile):
                print p, ' configuration file missing.'
                continue
            dfile = os.path.join(p, data_name)
            vis_slab2d(pfile, dfile, is_save)
            print pfile, dfile
            v = get_var(pfile)
            var.append(v)
            mat = loadmat(dfile)
            F.append(mat['F'][-1, 0])
            print v, mat['F'][-1, 0]

    savemat(os.path.join(path, 'data'), {'v': var, 'F': F})


def get_var(param_file):
    '''
        Get the main batch variable and its current value.
    '''
    cfg = SafeConfigParser(allow_no_value=True)
    cfg.optionxform = str
    cfg.read(param_file)
    section = cfg.get('Batch', 'section')
    # name list of the batch variable
    batch_var = json.loads(cfg.get('Batch', 'var'))
    var_name = batch_var[0]  # the main batch variable is the first one
    if (var_name == 'BC_coefficients_left'
            or var_name == 'BC_coefficients_right'):
        bc = json.loads(cfg.get(section, var_name))
        var = bc[1]
    else:
        var = cfg.getfloat(section, var_name)
    return var


def get_final_datafile(namelist):
    '''
    Each name has the form 'scft_out_XXXX.mat', where XXXX is a number.
    '''
    data = ''
    num = 0  # a number to be compared
    for f in namelist:
        name, ext = os.path.splitext(f)  # split into 'scft_out_XXXX', '.mat'
        fragments = name.split('_')  # split into 'scft', 'out', 'XXXX'
        n = int(fragments[-1])
        if n > num:
            num = n
            data = name
    return data


def vis_slab2d(param='param.ini', data='scft_out', is_save=False):
    '''
    Visualize 2D data generated by DiskXX, here XX represents the polymer model, e.g. XX = AB stands for A-B diblock copolymers.
    '''
    is_show = not is_save
    path = os.path.dirname(data)

    config = SCFTConfig.from_file(param)
    Nx, Ny = config.grid.Lx, config.grid.Ly
    Lx = config.uc.a
    Ly = config.uc.b
    print Nx, Ny, Lx, Ly
    Nxp = Nx + 1
    Nyp = 2 * Ny

    mat = loadmat(data)
    phiA = mat['phiA']
    phiB = mat['phiB']
    #phiAB = phiA - phiB

    if not (Nx, Ny) == phiA.shape:
        raise 'Data file does not match param file.'

    # Periodic in x direction, Fourier
    xxp = np.linspace(0, Lx, Nxp)
    # Non-periodic in y direction, Chebyshev
    #ii = np.arange(Ny)
    #yy = np.cos(np.pi * ii / (Ny-1))  # rr [-1, 1]
    yyp = np.linspace(0, Ly, Nyp)
    yp, xp = np.meshgrid(yyp, xxp)

    phiAp = np.zeros([Nxp, Ny])
    phiBp = np.zeros([Nxp, Ny])
    phiAp[:-1, :] = phiA
    phiAp[-1, :] = phiA[0, :]
    phiBp[:-1, :] = phiB
    phiBp[-1, :] = phiB[0, :]
    phiAp = cheb_interp2d_y(phiAp, yyp)
    phiBp = cheb_interp2d_y(phiBp, yyp)
    phiABp = phiBp - phiAp
    if is_show:
        plt.plot(yyp, phiAp[Nxp/2, :])
        plt.plot(yyp, phiBp[Nxp/2, :])
        plt.plot(yyp, phiABp[Nxp/2, :])
        #scft_contourf(xp, yp, phiAp, show_cbar=True)
        #scft_contourf(xp, yp, phiBp, show_cbar=True)
        #scft_contourf(xp, yp, phiABp, show_cbar=True)
        scft_contourf(xp, yp, phiAp)
        scft_contourf(xp, yp, phiBp)
        scft_contourf(xp, yp, phiABp)
        plt.show()
    if is_save:
        figA = os.path.join(path, 'phiA.png')
        figB = os.path.join(path, 'phiB.png')
        figAB = os.path.join(path, 'phiAB.png')
        scft_contourf(xp, yp, phiAp)
        plt.savefig(figA)
        scft_contourf(xp, yp, phiBp)
        plt.savefig(figB)
        scft_contourf(xp, yp, phiABp)
        plt.savefig(figAB)


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


if __name__ == '__main__':
    if args.batch:
        batch_vis_slab2d(args.path, args.param_file, args.data_file)
    else:
        vis_slab2d(args.param_file, args.data_file, args.save)
