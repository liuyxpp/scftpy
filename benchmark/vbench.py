#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vbench
======

A script for visulizing benchmark results.

Copyright (C) 2012 Yi-Xin Liu

"""

import argparse
import os.path

import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import mpltex.acs

parser = argparse.ArgumentParser(description='vbrush options')
parser.add_argument('-c', '--config',
                    default='param.ini',
                    help='the configuration file of polyorder.')

args = parser.parse_args()

Ns = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
ts_etdrk4 = np.array([[4.506, 3.001, 0.059, 0.047], 
                      [4.698, 3.033, 0.062, 0.092],
                      [4.894, 3.246, 0.069, 0.179],
                      [5.635, 3.667, 0.105, 0.356],
                      [6.993, 4.376, 0.288, 0.711],
                      [10.917, 6.254, 1.593, 1.654],
                      [87.592, 67.316, 14.658, 3.444],
                      [432.124, 379.615, 41.143, 6.676],
                      [1885.805, 1643.120, 212.187, 13.500],
                      [5307.210, 4185.168, 1025.591, 27.322]
                     ])

ts_oscheb = np.array([[30.437, 28.752, 0.017, 0.048], 
                      [40.478, 39.072, 0.017, 0.091],
                      [58.174, 56.475, 0.018, 0.182],
                      [97.606, 95.593, 0.017, 0.356],
                      [168.186, 165.795, 0.017, 0.711],
                      [316.814, 313.748, 0.017, 1.420],
                      [672.478, 666.718, 0.019, 3.357],
                      [1336.219, 1324.754, 0.020, 6.653],
                      [2624.528, 2596.846, 0.021, 13.344],
                      [7657.320, 7504.040, 0.025, 39.155]
                     ])

def plot_N():
    fig_name = 'bench_Ns200'

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(Ns, ts_oscheb[:,0], 'bs-', mew=0, label='OSCHEB total')
    ax.plot(Ns, ts_oscheb[:,1], 'bs--', mew=0, label='OSCHEB MDE')
    ax.plot(Ns, ts_oscheb[:,2], 'bs:', mew=0, label='OSCHEB init')
    #ax.plot(Ns, ts_oscheb[:,3], 'bs-.', mew=0, label='OSCHEB density')
    ax.plot(Ns, ts_etdrk4[:,0], 'ro-', mew=0, label='ETDRK4 total')
    ax.plot(Ns, ts_etdrk4[:,1], 'ro--', mew=0, label='ETDRK4 MDE')
    ax.plot(Ns, ts_etdrk4[:,2], 'ro:', mew=0, label='ETDRK4 init')
    #ax.plot(Ns, ts_etdrk4[:,3], 'ro-.', mew=0, label='ETDRK4 density')

    plt.xlabel('$N$')
    plt.ylabel('Computation time')
    plt.xscale('log')
    plt.yscale('log')
    plt.axis([2, 4000, 0.01, 1000000])
    ax.legend(loc='upper left')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


def plot_error_residual():
    base_str = '_error_N32'
    data_name = 'scft_out.mat'
    fig_name = 'SCFT' + base_str

    mat_oscheb = loadmat(os.path.join('oscheb'+base_str, data_name))
    ts_oscheb = mat_oscheb['time']
    errs_oscheb = mat_oscheb['err_residual']
    mat_etdrk4 = loadmat(os.path.join('etdrk4'+base_str, data_name))
    ts_etdrk4 = mat_etdrk4['time']
    errs_etdrk4 = mat_etdrk4['err_residual']

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(ts_oscheb[::10], errs_oscheb[::10], 
            'bs-', mew=0, label='OSCHEB')
    ax.plot(ts_etdrk4[::15], errs_etdrk4[::15], 
            'ro-', mew=0, label='ETDRK4')

    plt.xlabel('Computation time')
    plt.ylabel('Residual error')
    #plt.xscale('log')
    plt.yscale('log')
    #plt.axis([2, 4000, 0.01, 1000000])
    ax.legend(loc='upper right')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


def read_F(file):
    if not os.path.isfile(file):
        print file, 'hase been ignored.'
        return None, None
    mat = loadmat(file)
    F = mat['F']
    F.shape = (F.size,)
    time = mat['time']
    return F[-1], time[-1]

def plot_error_F():
    base_str = ''
    data_name = 'scft_out.mat'
    fig_name = 'SCFT_error_F' + base_str
    N = [32, 64, 128, 256, 512, 1024, 2048]
    F_etdrk4 = []
    time_etdrk4 = []
    N_etdrk4 = []
    F_oscheb = []
    time_oscheb = []
    N_oscheb = []
    for i in N:
        dir = 'etdrk4_error_N' + str(i) + base_str
        file = os.path.join(dir, data_name)
        F, time = read_F(file)
        if F is not None:
            F_etdrk4.append(F)
            time_etdrk4.append(time)
            N_etdrk4.append(i)

        dir = 'oscheb_error_N' + str(i) + base_str
        file = os.path.join(dir, data_name)
        F, time = read_F(file)
        if F is not None:
            F_oscheb.append(F)
            time_oscheb.append(time)
            N_oscheb.append(i)

    print F_etdrk4
    print F_oscheb

    savemat(fig_name, {'time_oscheb': time_oscheb, 'F_oscheb': F_oscheb,
                       'N_oscheb': N_oscheb, 'N_etdrk4': N_etdrk4,
                       'time_etdrk4': time_etdrk4, 'F_etdrk4': F_etdrk4})

    plt.figure()
    ax = plt.subplot(111)
    if time_oscheb:
        ax.plot(time_oscheb[:], F_oscheb[:]-F_oscheb[-1],
            'bs-', mew=0, label='OSCHEB CKE')
    if time_etdrk4:
        ax.plot(time_etdrk4[:], F_etdrk4[:]-F_etdrk4[-1], 
            'ro-', mew=0, label='ETDRK4 CKE')

    plt.xlabel('Computation time')
    plt.ylabel('$Error in free energy$')
    plt.xscale('log')
    #plt.yscale('log')
    #plt.axis([2, 4000, 0.01, 1000000])
    ax.legend(loc='lower right')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


def plot_error_F_all():
    ''' Data file are generated by plot_error_F. '''
    fig_name = 'SCFT_error_F'
    file_cke = 'SCFT_error_F'
    file_cke_Ns2000 = 'SCFT_error_F_Ns2000'
    file_heav = 'SCFT_error_F_Ns2000_heav'
    file_gauss = 'SCFT_error_F_Ns2000_gauss'

    mat_cke = loadmat(file_cke)
    time_oscheb = mat_cke['time_oscheb']
    N_oscheb = mat_cke['N_oscheb']
    F_oscheb = mat_cke['F_oscheb']
    time_etdrk4 = mat_cke['time_etdrk4']
    N_etdrk4 = mat_cke['N_etdrk4']
    F_etdrk4 = mat_cke['F_etdrk4']

    mat_cke_Ns2000 = loadmat(file_cke_Ns2000)
    time_oscheb_Ns2000 = mat_cke_Ns2000['time_oscheb']
    N_oscheb_Ns2000 = mat_cke_Ns2000['N_oscheb']
    F_oscheb_Ns2000 = mat_cke_Ns2000['F_oscheb']
    time_etdrk4_Ns2000 = mat_cke_Ns2000['time_etdrk4']
    N_etdrk4_Ns2000 = mat_cke_Ns2000['N_etdrk4']
    F_etdrk4_Ns2000 = mat_cke_Ns2000['F_etdrk4']

    mat_heav = loadmat(file_heav)
    time_oscheb_heav = mat_heav['time_oscheb']
    N_oscheb_heav = mat_heav['N_oscheb']
    F_oscheb_heav = mat_heav['F_oscheb']
    time_etdrk4_heav = mat_heav['time_etdrk4']
    N_etdrk4_heav = mat_heav['N_etdrk4']
    F_etdrk4_heav = mat_heav['F_etdrk4']

    mat_gauss = loadmat(file_gauss)
    time_oscheb_gauss = mat_gauss['time_oscheb']
    N_oscheb_gauss = mat_gauss['N_oscheb']
    F_oscheb_gauss = mat_gauss['F_oscheb']
    time_etdrk4_gauss = mat_gauss['time_etdrk4']
    N_etdrk4_gauss = mat_gauss['N_etdrk4']
    F_etdrk4_gauss = mat_gauss['F_etdrk4']

    plt.figure()
    ax = plt.subplot(111)
    if time_oscheb.size:
        ax.plot(time_oscheb[:], F_oscheb[:]-F_oscheb[-1],
                'bo-', mew=0, label='OSCHEB CKE')
    if time_oscheb_Ns2000.size:
        ax.plot(time_oscheb_Ns2000[:],
                F_oscheb_Ns2000[:]-F_oscheb_Ns2000[-1],
                'bv--', mew=0, label='OSCHEB CKE 2K')
    if time_oscheb_heav.size:
        ax.plot(time_oscheb_heav[:], F_oscheb_heav[:]-F_oscheb_heav[-1],
                'bs:', mew=0, label='OSCHEB Heaviside')
    if time_oscheb_gauss.size:
        ax.plot(time_oscheb_gauss[:],
                F_oscheb_gauss[:]-F_oscheb_gauss[-1],
                'b^-.', mew=0, label='OSCHEB Gaussian')
    if time_etdrk4.size:
        ax.plot(time_etdrk4[:], F_etdrk4[:]-F_etdrk4[-1],
                'ro-', mew=0, label='ETDRK4 CKE')
    if time_etdrk4_Ns2000.size:
        ax.plot(time_etdrk4_Ns2000[:],
                F_etdrk4_Ns2000[:]-F_etdrk4_Ns2000[-1],
                'rv--', mew=0, label='ETDRK4 CKE 2K')
    if time_etdrk4_heav.size:
        ax.plot(time_etdrk4_heav[:], F_etdrk4_heav[:]-F_etdrk4_heav[-1],
                'rs:', mew=0, label='ETDRK4 Heaviside')
    if time_etdrk4_gauss.size:
        ax.plot(time_etdrk4_gauss[:],
                F_etdrk4_gauss[:]-F_etdrk4_gauss[-1],
                'r^-.', mew=0, label='ETDRK4 Gaussian')

    plt.xlabel('Computation time')
    plt.ylabel('$Error in free energy$')
    plt.xscale('log')
    #plt.yscale('log')
    #plt.axis([2, 4000, 0.01, 1000000])
    ax.legend(loc='lower right')
    plt.savefig(fig_name+'_time', bbox_inches='tight')
    plt.show()

    plt.figure()
    ax = plt.subplot(111)
    if N_oscheb.size:
        ax.plot(N_oscheb[:], F_oscheb[:]-F_oscheb[-1],
                'bo-', mew=0, label='OSCHEB CKE')
    if N_oscheb_Ns2000.size:
        ax.plot(N_oscheb_Ns2000[:],
                F_oscheb_Ns2000[:]-F_oscheb_Ns2000[-1],
                'bv--', mew=0, label='OSCHEB CKE 2K')
    if N_oscheb_heav.size:
        ax.plot(N_oscheb_heav[:], F_oscheb_heav[:]-F_oscheb_heav[-1],
                'bs:', mew=0, label='OSCHEB Heaviside')
    if N_oscheb_gauss.size:
        ax.plot(N_oscheb_gauss[:],
                F_oscheb_gauss[:]-F_oscheb_gauss[-1],
                'b^-.', mew=0, label='OSCHEB Gaussian')
    if N_etdrk4.size:
        ax.plot(N_etdrk4[:], F_etdrk4[:]-F_etdrk4[-1],
                'ro-', mew=0, label='ETDRK4 CKE')
    if N_etdrk4_Ns2000.size:
        ax.plot(N_etdrk4_Ns2000[:],
                F_etdrk4_Ns2000[:]-F_etdrk4_Ns2000[-1],
                'rv--', mew=0, label='ETDRK4 CKE 2K')
    if N_etdrk4_heav.size:
        ax.plot(N_etdrk4_heav[:], F_etdrk4_heav[:]-F_etdrk4_heav[-1],
                'rs:', mew=0, label='ETDRK4 Heaviside')
    if N_etdrk4_gauss.size:
        ax.plot(N_etdrk4_gauss[:],
                F_etdrk4_gauss[:]-F_etdrk4_gauss[-1],
                'r^-.', mew=0, label='ETDRK4 Gaussian')

    plt.xlabel('$N_z$')
    plt.ylabel('$Error in free energy$')
    plt.xscale('log')
    #plt.yscale('log')
    #plt.axis([2, 4000, 0.01, 1000000])
    ax.legend(loc='lower right')
    plt.savefig(fig_name+'_Nz', bbox_inches='tight')
    plt.show()

def plot_x0():
    base_str = ''
    data_name = 'scft_out.mat'
    N = [32, 64, 128, 256, 512, 1024, 2048]
    x0_etdrk4 = []
    x0_oscheb = []
    for i in N:
        dir_etdrk4 = 'etdrk4_error_N' + str(i) + base_str
        mat_etdrk4 = loadmat(os.path.join(dir_etdrk4, data_name))
        x = mat_etdrk4['x']
        ix0_etdrk4 = mat_etdrk4['ix0']
        x0_etdrk4.append(x[ix0])
        dir_oscheb = 'oscheb_error_N' + str(i) + base_str
        mat_oscheb = loadmat(os.path.join(dir_oscheb, data_name))
        x = mat_oscheb['x']
        ix0 = mat_oscheb['ix0']
        x0_oscheb.append(x[ix0])

    fig_name = 'SCFT_error_x0'
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(N, x0_oscheb, 'bs-', mew=0, label='OSCHEB')
    ax.plot(N, x0_etdrk4, 'ro-', mew=0, label='ETDRK4')

    plt.xlabel('$N_z$')
    plt.ylabel('$x_0$')
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.axis([2, 4000, 0.01, 1000000])
    ax.legend(loc='upper right')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


def plot_error_F_ka():
    base_dir = 'etdrk4_error'
    base_str = 'Nz_Ns200_sigma0.5_ka4n'
    data_name = 'scft_out.mat'
    fig_name = 'SCFT_error_F_ka' + base_str
    N = [64, 128, 256, 512]
    F_etdrk4 = []
    time_etdrk4 = []
    N_etdrk4 = []
    for i in N:
        file = os.path.join(base_dir, base_str, str(i), data_name)
        F, time = read_F(file)
        if F is not None:
            F_etdrk4.append(F)
            time_etdrk4.append(time)
            N_etdrk4.append(i)


    print F_etdrk4

    savemat(fig_name, {'N_etdrk4': N_etdrk4,
                       'time_etdrk4': time_etdrk4, 'F_etdrk4': F_etdrk4})

    plt.figure()
    ax = plt.subplot(111)
    if time_etdrk4:
        ax.plot(time_etdrk4[:], F_etdrk4[:]-F_etdrk4[-1], 
            'ro-', mew=0, label='ETDRK4 CKE')

    plt.xlabel('Computation time')
    plt.ylabel('$Error in free energy$')
    plt.xscale('log')
    #plt.yscale('log')
    #plt.axis([2, 4000, 0.01, 1000000])
    ax.legend(loc='lower right')
    plt.savefig(fig_name+'.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    #plot_N()
    #plot_error_residual()
    #plot_error_F()
    #plot_error_F_all()
    plot_error_F_ka()
