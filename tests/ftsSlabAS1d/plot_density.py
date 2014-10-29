# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use('Agg')  # To avoid launching interactive plot, such as wxAgg.
import matplotlib.pyplot as plt

import mpltex
from mpltex.styles import markers, _colors

import os.path
import itertools
import argparse
from ConfigParser import SafeConfigParser
import numpy as np
from scipy.io import loadmat, savemat

from scftpy import list_datafile
from chebpy import cheb_interpolation_1d, cheb_quadrature_clencurt

parser = argparse.ArgumentParser(description='plot_density options')

parser.add_argument('-p','--phi',
                    action='store_true',
                    help= 'If present or True, \
                           plot density profile.')
parser.add_argument('-w','--iw',
                    action='store_true',
                    help= 'If present or True, \
                           plot <iw> profile.')
parser.add_argument('-f','--force',
                    action='store_true',
                    help= 'If present or True, \
                           plot C*\phi-<iw>/B profile.')

args = parser.parse_args()

@mpltex.acs_decorator
def get_data(path, prefix):
    datafiles = list_datafile(path, prefix)
    Nxs = []
    dxs = []
    Fs = []
    for dfile in datafiles:
        p = os.path.dirname(dfile)
        label = os.path.basename(p)
        print label
        mat = loadmat(dfile)


@mpltex.acs_decorator
def plot_density():
    #base_dir = 'benchmark/BCC'
    data_dir = ['B0.5_C25_scft/scft_out_943.mat',
                'B0.5_C25_fts/fts_out_50000.mat',
                'B25_C0.5_scft/scft_out_788.mat',
                'B25_C0.5_fts/fts_out_50000.mat',
                ]
    is_cl = [False, True, False, True]
    labels = ['SCFT, $B=0.5, C=25$',
              'CL, $B=0.5, C=25$',
              'SCFT, $B=25, C=0.5$',
              'CL, $B=25, C=0.5$',
              ]

    fig, ax = plt.subplots(1)
    linestyle = mpltex.linestyle_generator(lines=['-'], markers=[])

    i = 0
    for f in data_dir:
        print 'Processing ', f
        try:
            mat = loadmat(f)
        except:
            print 'Missing datafile', f
            continue
        x = mat['x']
        x = x.reshape(x.size)
        if is_cl[i]:
            phi = mat['phi_avg'].real
            phi = phi.reshape(phi.size)
        else:
            phi = mat['phi']
            phi = phi.reshape(phi.size)
        y = np.arange(-1, 1, 0.01)
        phi = cheb_interpolation_1d(y, phi)
        yp = 0.5 * (y + 1) * (x.max() - x.min())
        ax.plot(yp, phi, label=labels[i], **linestyle.next())
        i += 1

    #ax.set_yscale('log')
    ax.locator_params(nbins=5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\phi$')
    #ax.set_xlim([0.4, 0.6])
    #ax.set_ylim([1.71, 1.76])
    ax.legend(loc='best')

    fig.tight_layout(pad=0.1)
    fig.savefig('density_profile')


@mpltex.acs_decorator
def plot_iw():
    #base_dir = 'benchmark/BCC'
    data_dir = [#'B0.5_C25_fts_fixphi_run1/fts_out_50000.mat',
                'B0.5_C25_fts_fixscftphi/fts_out_100000.mat',
                #'B0.5_C25_fts_fixscftphi_run1/fts_out_50000.mat',
                #'B0.5_C25_fts_fixscftphi_run2/fts_out_50000.mat',
                #'B0.5_C25_fts_fixscftphi_run3/fts_out_100000.mat',
                'B0.5_C25_fts_fixscftphi_run4/fts_out_100000.mat',
                'B0.5_C25_fts_fixscftphi_run5/fts_out_100000.mat',
                #'B0.5_C25_fts_fixscftphi_run6/fts_out_100000.mat',
                #'B0.5_C25_fts_fixotherphi/fts_out_50000.mat',
                #'B0.5_C25_fts_fixphi_Lx128/fts_out_50000.mat',
                ]
    is_cl = [True, True, True, True, True]
    labels = [#'$\phi_{CL}, L_x=64$',
              '$\phi_{SCFT}, \lambda\Delta t=10^{-6}, t=8 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-3}, t=3 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-4}, t=3 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-4}, t=8 \\times 10^4$',
              '$\phi_{SCFT}, \lambda\Delta t=10^{-5}, t=8 \\times 10^4$',
              '$\phi_{SCFT}, \lambda\Delta t=10^{-5}, t=8 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-2}, t=8 \\times 10^4$',
              #'$\phi_{other}, L_x=64$',
              #'$\phi_{CL}, L_x=128$'
              ]
    B = [0.5, 0.5, 0.5, 0.5, 0.5]
    C = [25, 25, 25, 25, 25]

    fig, ax = plt.subplots(1)
    linestyle = mpltex.linestyle_generator(lines=['-'], markers=['-o'],
                                           hollow_styles=[])

    i = 0
    for f in data_dir:
        print 'Processing ', f
        try:
            mat = loadmat(f)
        except:
            print 'Missing datafile', f
            continue
        x = mat['x']
        x = x.reshape(x.size)
        if is_cl[i]:
            iw_avg = mat['iw_avg'].real
            iw_avg = iw_avg.reshape(iw_avg.size)
            phi_avg = mat['phi_avg'].real
            phi_avg = phi_avg.reshape(phi_avg.size)
        y = np.arange(-1, 1, 0.01)
        #iw_avg = cheb_interpolation_1d(y, iw_avg)
        phi_avg = cheb_interpolation_1d(y, phi_avg)
        yp = 0.5 * (y + 1) * (x.max() - x.min())
        ax.plot(x, iw_avg, label=labels[i], **linestyle.next())
        i += 1

    #ax.set_yscale('log')
    ax.locator_params(nbins=5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$<iw>$')
    #ax.set_xlim([0.4, 0.6])
    #ax.set_ylim([1.71, 1.76])
    ax.legend(loc='best')

    fig.tight_layout(pad=0.1)
    fig.savefig('iw_profile')


@mpltex.acs_decorator
def plot_force():
    #base_dir = 'benchmark/BCC'
    data_dir = [#'B0.5_C25_fts_fixphi_run1/fts_out_50000.mat',
                #'B0.5_C25_fts_fixscftphi/fts_out_100000.mat',
                #'B0.5_C25_fts_fixscftphi_run1/fts_out_50000.mat',
                #'B0.5_C25_fts_fixscftphi_run2/fts_out_50000.mat',
                #'B0.5_C25_fts_fixscftphi_run3/fts_out_100000.mat',
                #'B0.5_C25_fts_fixscftphi_run4/fts_out_100000.mat',
                #'B0.5_C25_fts_fixscftphi_run5/fts_out_100000.mat',
                #'B0.5_C25_fts_fixscftphi_run6/fts_out_100000.mat',
                #'B0.5_C25_fts_fixscftphi_run7/fts_out_100000.mat',
                'B25_C0.5_fts_fixclphi_update_phi_imag/fts_out_100000.mat',
                #'B25_C0.5_fts_fixphi/fts_out_50000.mat',
                #'B25_C0.5_fts_fixphi_run1/fts_out_50000.mat',
                #'B25_C0.5_fts_fixscftphi/fts_out_100000.mat',
                #'B25_C0.5_fts_fixscftphi_update_phi_imag/fts_out_150000.mat',
                #'B25_C0.5_fts_fixscftphi_update_phi_imag_run1/fts_out_100000.mat',
                #'B25_C0.5_fts_fixscftphi_update_phi_imag_run2/fts_out_100000.mat',
                #'B0.5_C25_fts_fixscftphi_update_phi_imag_run1/fts_out_100000.mat',
                #'B0.5_C25_fts_fixscftphi_update_phi_imag_run2/fts_out_100000.mat',
                #'B0.5_C25_fts_fixscftphi_update_phi_imag_run3/fts_out_200000.mat',
                #'B0.5_C25_fts_fixotherphi/fts_out_50000.mat',
                #'B0.5_C25_fts_fixphi_Lx128/fts_out_50000.mat',
                ]
    is_cl = [True, True, True, True, True]
    labels = [#'$\phi_{CL}, L_x=64$',
              '$\phi_{CL}, \lambda\Delta t=10^{-3}, t=8 \\times 10^4$',
              #'$\phi_{CL}, \lambda\Delta t=10^{-3}, t=3 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-3}, t=8 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-3}, t=1 \\times 10^5$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-3}, t=8 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-3}, t=6 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-4}, t=5 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-3}, t=8 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-3}, t=1.8 \\times 10^5$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-3}, t=3 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-4}, t=3 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-4}, t=8 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-5}, t=8 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-5}, t=8 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-2}, t=8 \\times 10^4$',
              #'$\phi_{SCFT}, \lambda\Delta t=10^{-6}, t=8 \\times 10^4$',
              #'$\phi_{other}, L_x=64$',
              #'$\phi_{CL}, L_x=128$'
              ]
    B = [0.5, 0.5, 0.5, 0.5, 0.5]
    C = [25, 25, 25, 25, 25]

    fig, ax = plt.subplots(1)
    linestyle = mpltex.linestyle_generator(lines=['-'], markers=['o'],
                                           hollow_styles=[])

    i = 0
    for f in data_dir:
        print 'Processing ', f
        try:
            mat = loadmat(f)
        except:
            print 'Missing datafile', f
            continue
        x = mat['x']
        x = x.reshape(x.size)
        if is_cl[i]:
            iw_avg = mat['iw_avg'].real
            iw_avg = iw_avg.reshape(iw_avg.size)
            phi = mat['phi'].real
            phi = phi.reshape(phi.size)
        y = np.arange(-1, 1, 0.01)
        force = C[i] * phi - iw_avg / B[i]
        print "\tmean force: ", 0.5 * cheb_quadrature_clencurt(force)
        #force = cheb_interpolation_1d(y, force)
        #yp = 0.5 * (y + 1) * (x.max() - x.min())
        ax.plot(x, force, label=labels[i], **linestyle.next())
        i += 1

    #ax.set_yscale('log')
    ax.locator_params(nbins=5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$<\\frac{\delta H}{\delta \phi}>_{\phi}$')
    #ax.set_xlim([0.4, 0.6])
    #ax.set_ylim([1.71, 1.76])
    ax.legend(loc='best')

    fig.tight_layout(pad=0.1)
    fig.savefig('force_profile')


if __name__ == '__main__':
    if args.phi:
        plot_density()
    if args.iw:
        plot_iw()
    if args.force:
        plot_force()


