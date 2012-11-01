# -*- coding: utf-8 -*-
"""
brush
=====

SCFT for polymer melt brush.

References
----------

Liu, Y. X. *Polymer Brush III*, Technical Reports, Fudan University, 2012. 

"""

import os

import numpy as np
from scipy.integrate import simps, trapz, romb
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

import mpltex.acs
from chebpy import cheb_mde_neumann_etdrk4, cheb_mde_dirichlet_etdrk4
from chebpy import cheb_mde_robin_etdrk4, cheb_mde_robin_dirichlet_etdrk4
from chebpy import cheb_mde_robin_neumann_etdrk4
from chebpy import clencurt_weights_fft, cheb_quadrature_clencurt 
from chebpy import cheb_D1_mat

from scftpy import SCFTConfig
from scftpy import quad_open4, quad_semiopen4, quad_semiopen3

__all__ = ['Brush',
          ]

def solve_mde(w, q, L, Ns, ds):
    #cheb_mde_neumann_etdrk4(w, q[0], L, Ns, ds, q)
    cheb_mde_dirichlet_etdrk4(w, q[0], L, Ns, ds, q)
    #cheb_mde_robin_dirichlet_etdrk4(w, q[0], L, Ns, -0.5*L, ds, q)
    #cheb_mde_robin_neumann_etdrk4(w, q[0], L, Ns, -.5*L, ds, q)
    #cheb_mde_robin_etdrk4(w, q[0], L, Ns, -0.5*L, 1.5*L, ds, q)


def generate_IC(w, ds, x0, L):
    '''
    Generate Initial Condition for MDE. 
    '''
    N = w.size - 1
    delta, x, ix0 = make_delta(N, x0, L)
    #u0 = delta
    u0 = np.exp(-ds*w) / np.sqrt(4*np.pi*ds) \
            * np.exp(-(x[:,0]-x0)**2/(4*ds))

    return u0, x, ix0


def make_delta(N, x0, L):
    '''
    Construct a delta function delta(z-z0) on Chebyshev grid.
    Approximate Dirac delta funcition by differentiating a step function.
    '''
    ix = int(np.arccos(2*x0/L-1) / np.pi * N)
    D, x = cheb_D1_mat(N)
    H = np.zeros(N+1)
    H[0:ix] = 1.
    H[ix] = .5
    u = (2/L) * np.dot(D, H)

    #u = np.zeros(N+1)
    #ii = np.arange(N+1)
    #x = np.cos(np.pi * ii / N)
    #w = clencurt_weights_fft(N)
    #ix = N - 1 # set the second-to-last element to be the delta postion
    #u[ix] = (2.0/L) / w[ix]

    x = .5 * (x + 1) * L

    return u, x, ix


def calc_density(q, qc):
    qqc = qc * q[::-1,:]
    Ms, Lx = q.shape
    ds = 1. / (Ms - 1)
    phi = np.zeros(Lx)
    for i in xrange(Lx):
        #phi[i] = simps(qqc[:,i], dx=ds)
        #phi[i] = trapz(qqc[:,i], dx=ds)
        #phi[i] = quad_open4(qqc[:,i], dx=ds)
        #phi[i] = quad_semiopen3(qqc[:,i], dx=ds)
        phi[i] = quad_semiopen4(qqc[:,i], dx=ds)

    return phi


def calc_brush_height(x, phi):
    '''
    The brush height is defined as:
        h = <x> = \int_{x*phi(x)dx}/\int_{phi(x)dx}
    '''
    I1 = cheb_quadrature_clencurt(x[:,0]*phi)
    I2 = cheb_quadrature_clencurt(phi)
    return I1/I2


class Brush(object):
    def __init__(self, cfgfile):
        self.config = SCFTConfig.from_file(cfgfile)
        self.init()
        param_file = os.path.join(self.config.scft.base_dir, 
                               self.config.scft.param_file)
        self.config.save_to_mat(param_file)

    def init(self):
        self.build_model()
        self.build_grid()

    def build_model(self):
        config = self.config
        self.N = config.model.N
        self.a = config.model.a
        self.chiN = config.model.chiN

    def build_grid(self):
        config = self.config
        Lx = config.grid.Lx; Ly = config.grid.Ly; Lz = config.grid.Lz
        d = config.grid.dimension
        Ms = config.grid.Ms
        if d == 3:
            self.w = np.random.rand(Lx, Ly, Lz) - 0.5
            self.q = np.ones((Ms, Lx, Ly, Lz))
            self.qc = np.zeros((Ms, Lx, Ly, Lz))
        elif d == 2:
            self.w = np.random.rand(Lx, Ly) - 0.5
            self.q = np.ones((Ms, Lx, Ly))
            self.qc = np.zeros((Ms, Lx, Ly))
        elif d == 1:
            #self.w = np.random.rand(Lx)
            L = config.uc.a
            ups = config.model.excluded_volume
            N = Lx - 1
            ii = np.arange(N+1)
            x = np.cos(np.pi * ii / N)
            x = .5 * (x + 1) * L
            self.w = (1 - (.5*x/L)**2)
            self.q = np.zeros((Ms, Lx))
            self.q[0,:] = 1.
            self.qc = np.zeros((Ms, Lx))
        else:
            raise ValueError('Only 1D, 2D and 3D spaces are allowed!')

    def run(self):
        config = self.config
        x0 = 0.5 * np.sqrt(6.0/self.N[0])
        #x0 = 0
        L = config.uc.a
        Lx = config.grid.Lx
        delta, x, ix0 = make_delta(Lx-1, x0, L)
        print 'x0 =', x0, 'ix0 =', ix0
        Ms = config.grid.Ms
        ds = 1. / (Ms - 1)
        sigma = config.model.graft_density
        ups = config.model.excluded_volume
        lam = config.grid.lam[0]
        x_ref = (4. * ups * sigma) ** (1./3)
        x_rescaled = x / x_ref
        phi = np.zeros(Lx)
        phi_ref = (0.25 * sigma * sigma / ups) ** (1./3)
        beta = 0.25 * x_ref * x_ref
        print 'sigma =', sigma, 'ups =', ups, 'beta =', beta 
        print 'Lx =', Lx, 'Ms =', Ms, 'lam =', lam
        print 'x_ref =', x_ref, 'phi_ref =', phi_ref
        display_interval = config.scft.display_interval
        record_interval = config.scft.record_interval
        save_interval = config.scft.save_interval
        data_file = os.path.join(config.scft.base_dir,
                                 config.scft.data_file)
        ts = []
        Fs = []
        hs = []
        errs_residual = []
        errs_phi = []
        for t in xrange(1, config.scft.max_iter+1):
            #raw_input()

            # Solve MDE
            if t % display_interval == 0:
                plt.plot(x_rescaled, self.w)
                plt.ylabel('w')
                plt.show()
            #    plt.plot(x_rescaled, self.q[0])
            #    plt.ylabel('q(0)')
            #    plt.show()
            self.q[0,:] = 1.
            solve_mde(self.w, self.q, L, Ms, ds)
            if t % display_interval == 0:
                plt.plot(x_rescaled, self.q[-1])
                plt.ylabel('q(-1)')
                plt.show()
            self.qc[0] = (sigma / self.q[-1,ix0]) * delta
            qc1, x, ix0 = generate_IC(self.w, ds, x0, L)
            self.qc[1] = (sigma / self.q[-1,ix0]) * qc1
            if t % display_interval == 0:
                plt.plot(x_rescaled, self.qc[0])
                plt.ylabel('qc(0)')
                plt.show()
            solve_mde(self.w, self.qc[1:,:], L, Ms-1, ds)
            if t % display_interval == 0:
                plt.plot(x_rescaled, self.qc[-1])
                plt.ylabel('qc(-1)')
                plt.show()

            # Calculate energy
            Q = self.q[-1,ix0]
            #Q = 1.0
            #F1 = -0.5 * cheb_quadrature_clencurt(self.w*self.w)
            #F2 = -sigma * np.log(Q)
            F1 = -0.5 * beta * 0.5 * (L/x_ref) \
                    * cheb_quadrature_clencurt((phi/phi_ref)**2)
            F2 = -np.log(Q)
            F = F1 + F2
            
            # Calculate density
            phi0 = phi
            phi = calc_density(self.q, self.qc)
            if t % display_interval == 0:
                plt.plot(x_rescaled, phi / phi_ref)
                plt.ylabel('$\phi$')
                plt.show()

            # Estimate error
            res = phi - self.w / ups
            #res = phi - phi0
            if t % display_interval == 0:
                plt.plot(x_rescaled, res / phi_ref)
                plt.ylabel('res')
                plt.show()
            err1 = np.linalg.norm(res)
            err2 = np.linalg.norm(phi-phi0)

            h = calc_brush_height(x/x_ref, phi/phi_ref)

            if t % record_interval == 0:
                ts.append(t)
                Fs.append(F)
                hs.append(h)
                errs_residual.append(err1)
                errs_phi.append(err2)
                print t, '\t', F1, '\t', F2, '\t', F, '\t', h
                print '\t', err1, '\t', err2
            if t % save_interval == 0:
                savemat(data_file+'_'+str(t), {'t':ts, 'F':Fs, 'h':hs,
                                    'beta':beta, 'x_ref':x_ref,
                                    'phi_ref':phi_ref, 'x':x/x_ref,
                                    'err_residual':errs_residual,
                                    'err_phi':errs_phi,
                                    'phi':phi/phi_ref, 'w':self.w})

            # Update field
            self.w = self.w + lam * res * ups


