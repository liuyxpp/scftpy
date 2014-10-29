#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
force_matching
==============

A script for performing force matching field-theoretic (FT) complex Langevin (CL) simulations (FTS/CL) and its corresponding analytical approximations to construct a free energy functional containing only density functions for fluctuating polymer systems.

NOTE: Only 1D Chebyshev-Gauss-Lobatto grid is supported.

Revision: 2014.10.29.

Copyright (C) 2014 Yi-Xin Liu (lyx@fudan.edu.cn)

'''

import os
import argparse
import glob

import matplotlib as mpl
mpl.use('Agg')  # To avoid launching interactive plot, such as wxAgg.
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

import mpltex
from chebpy import cheb_quadrature_clencurt, cheb_D1_fchebt
from scftpy import SCFTConfig
from scftpy import list_datapath, get_data_file

parser = argparse.ArgumentParser(description='force_matching options')

parser.add_argument('-b', '--batch',
                    action='store_true',
                    help='If present or True, perform batch mode.')
parser.add_argument('-r', '--render',
                    action='store_true',
                    help='If present or True, render force.')
parser.add_argument('-s', '--basis_set',
                    default='GSD2',
                    help='Available basis set: GSD1, GSD2.')
parser.add_argument('-p', '--path',
                    default='.',
                    help='Path to be processed, for batch mode use.')
parser.add_argument('-d', '--data_file',
                    default='fts_out',
                    help='FTS/CL or SCFT generated data filename prefix.')
args = parser.parse_args()


def main():
    if args.batch:
        paths = []
        Ks = []
        for p in list_datapath(args.path, args.data_file):
            K = process_single_data(p, args.basis_set,
                                    args.redner, args.data_file)
            paths.append(p)
            Ks.append(K)
            print p
            print K
        savemat(os.path.join(args.path, 'force_fit_'+args.basis_set+'.mat'))

    else:
        K= process_single_data(args.path, args.basis_set,
                               args.render, args.data_file)
        print args.path
        print K


@mpltex.presentation_decorator
def plot_spatial(x, data, ylabel, labels, figname):
    fig, ax = plt.subplots(1)
    linestyle = mpltex.linestyle_generator(lines=['-'], markers=['o'],
                                           hollow_styles=[])
    i = 0
    for d in data:
        if len(data) > 1:
            ax.plot(x, d, label=labels[i], **linestyle.next())
        else:
            ax.plot(x, d, **linestyle.next())
        i += 1

    ax.locator_params(nbins=5)
    ax.set_xlabel('$z$')
    ax.set_ylabel(ylabel)
    if len(data) > 1:
        ax.legend(loc='best')

    fig.tight_layout(pad=0.1)
    fig.savefig(figname)
    plt.close(fig)


@mpltex.acs_decorator
def plot_timeseries(t, data, ylabel, figname):
    fig, ax = plt.subplots(1)
    linestyle = mpltex.linestyle_generator(lines=['-'], markers=[],
                                           hollow_styles=[])
    i = 0
    for d in data:
        if t.size > 0:
            ax.plot(t, d, **linestyle.next())
        else:
            ax.plot(d, **linestyle.next())
        i += 1

    ax.locator_params(nbins=5)
    ax.set_xlabel('$t$')
    ax.set_ylabel(ylabel)

    fig.tight_layout(pad=0.1)
    fig.savefig(figname)
    plt.close(fig)


def process_single_data(data_path, basis_type, render, prefix='fts_out'):
    pfile = os.path.join(data_path, 'param_out.mat')
    mat = loadmat(pfile)
    C = mat['f'][0,0]
    B = mat['chiN'][0,0]
    L = mat['a'][0,0]

    dfile = get_data_file(data_path, prefix)
    mat = loadmat(dfile)
    t = mat['t'][0,:]
    H = mat['F'][0,:].real
    mu = mat['mu'][0,:].real
    if render:
        figname = os.path.join(data_path, 'H.png')
        plot_timeseries(t, [H], '$H$', figname)
        figname = os.path.join(data_path, 'mu.png')
        plot_timeseries(t[t.size-mu.size:], [mu], '$\mu$', figname)

    x = mat['x'][0,:]
    phi = mat['phi'][0,:].real
    iw_avg = mat['iw_avg'][0,:].real
    mu_avg = mat['mu_avg'][0,0].real
    if render:
        figname = os.path.join(data_path, 'phi')
        plot_spatial(x, [phi], '$\phi$', [], figname)
        figname = os.path.join(data_path, 'iw_avg')
        plot_spatial(x, [iw_avg], '$<iw>$', [], figname)

    basis_set = generate_basis_set(basis_type, C, phi, L)
    target = generate_target(basis_type, C, phi, iw_avg, mu_avg)
    K = matching_force_by_linear_least_square(basis_set, target)
    f_sim = force_simulation(B, C, phi, iw_avg)
    f_fit = force_fit(B, C, phi, basis_type, basis_set, K, mu_avg)
    figname = os.path.join(data_path, 'forces_'+basis_type)
    plot_spatial(x, [f_sim, f_fit], '$\phi^2\\frac{\\delta H}{\\delta\phi}$',
                 ['CL', 'Fitted'], figname)

    if basis_type.upper() == 'GSD1':
        K_out = np.zeros(K.size+1)
        K_out[:-1] = K
        K_out[-1] = mu_avg
    elif basis_type.upper() == 'GSD2':
        K_out = K
    else:
        print 'Abort: unknown basis type ', basis_type, '.'
        exit(1)
    return K_out


def force_simulation(B, C, phi, iw_avg):
    return B * C**2 * phi**3 - C * iw_avg * phi**2


def force_fit(B, C, phi, basis_type, basis_set, K, mu_avg=0.0):
    if basis_type.upper() == 'GSD1':
        force = B * C**2 * phi**3 - C * mu_avg * phi**2
    elif basis_type.upper() == 'GSD2':
        force = B * C**2 * phi**3
    else:
        print 'Abort: unknown basis type ', basis_type, '.'
        exit(1)

    for i in xrange(len(basis_set)):
        force -= K[i] * basis_set[i]
    return force


def matching_force_by_linear_least_square(basis_set, target):
    '''
    Perform force matching by linear least square method.
        X * K = R
    where K is a vector of fitted coefficients, R is a vector, and X is a matrix. R and X can be constructed from the input basis and target.
    In particular,
        X_{ij} = \int dx basis[i]*basis[j]
        R_i = \int dx basis[i]*target
    '''
    n = len(basis_set)
    X = np.zeros( (n,n) )
    R = np.zeros(n)
    for i in xrange(n):
        R[i] = 0.5*cheb_quadrature_clencurt(basis_set[i] * target)
        for j in xrange(n):
            X[i,j] = 0.5*cheb_quadrature_clencurt(basis_set[i] * basis_set[j])
    print 'X =', X
    print 'R =', R
    return np.linalg.solve(X, R)


def generate_target(basis_type, C, phi, iw_avg, mu_avg):
    if basis_type.upper() == 'GSD1':
        target = C * (iw_avg - mu_avg) * phi**2
    elif basis_type.upper() == 'GSD2':
        target = C * iw_avg * phi**2
    else:
        print 'Abort: unknown basis type ', basis_type, '.'
        exit(1)
    return target


def basis_linear(C, phi):
    return C * phi


def basis_square(C, phi):
    return C * phi**2


def basis_triple(C, phi):
    return C * phi**3


def basis_laplacian_gsd(C, phi, L):
    '''
        Here, to avoid sigular at the boundary, we return
            C * \phi * Laplaican{phi}
        instead of
            C * \phi^{-1} * Laplaican{phi}
    '''
    dphi = (2.0 / L) * cheb_D1_fchebt(phi)
    d2phi = (2.0 / L) * cheb_D1_fchebt(dphi)
    return C * phi * d2phi


def basis_laplacian(C, phi, L):
    dphi = (2.0 / L) * cheb_D1_fchebt(phi)
    d2phi = (2.0 / L) * cheb_D1_fchebt(dphi)
    return C * d2phi


def basis_gradient_square_gsd(C, phi, L):
    '''
        Here, to avoid sigular at the boundary, we return
            C * \grad{phi}**2
        instead of
            C * \phi^{-2} * \grad{phi}**2
    '''
    dphi = (2.0 / L) * cheb_D1_fchebt(phi)
    return C * dphi**2


def basis_gradient_square(C, phi, L):
    dphi = (2.0 / L) * cheb_D1_fchebt(phi)
    return C * dphi**2


def generate_basis_set(basis_type, C, phi, L):
    basis_set = []
    if basis_type.upper() == 'GSD1':
        basis = 2 * basis_laplacian_gsd(C, phi, L)
        basis -= basis_gradient_square_gsd(C, phi, L)
        basis_set.append(basis)
    elif basis_type.upper() == 'GSD2':
        basis1 = 2 * basis_laplacian_gsd(C, phi, L)
        basis1 -= basis_gradient_square_gsd(C, phi, L)
        basis_set.append(basis1)
        basis2 = C * phi**2
        basis_set.append(basis2)
    else:
        print 'Abort: unknown basis type ', basis_type, '.'
        exit(1)
    return basis_set


if __name__ == '__main__':
    main()
