# -*- coding: utf-8 -*-
"""
brush
=====

SCFT for polymer brushes in good solvents.
All quantities are in dimensionless form.
Length is rescaled by \hat{z} = N(2w\sigma a^2/3)^(1/3).
Density is rescaled by \hat{\rho} = N\sigma/\hat{z}.
The only controlling parameter is \beta = (\hat{z}/2R_g)^2.

References
----------

Liu, Y. X. *Polymer Brushes III*, Technical Reports, Fudan University, 2012. 

"""

import os
from time import clock

import numpy as np
from scipy.integrate import simps, trapz, romb
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

import mpltex.acs
from chebpy import BC, ETDRK4, OSCHEB
from chebpy import DIRICHLET, NEUMANN, ROBIN
from chebpy import clencurt_weights_fft, cheb_quadrature_clencurt 
from chebpy import cheb_D1_mat

from scftpy import SCFTConfig
from scftpy import quad_open4, quad_semiopen4, quad_semiopen3

__all__ = ['Brush',
          ]

def solve_mde(w, q, L, Ns, ds):
    #cheb_mde_neumann_etdrk4(w, q[0], L, Ns, ds, q)
    #cheb_mde_dirichlet_etdrk4(w, q[0], L, Ns, ds, q)
    #cheb_mde_robin_dirichlet_etdrk4(w, q[0], L, Ns, -0.5*L, ds, q)
    #cheb_mde_robin_neumann_etdrk4(w, q[0], L, Ns, -.5*L, ds, q)
    #cheb_mde_robin_etdrk4(w, q[0], L, Ns, -0.5*L, 1.5*L, ds, q)
    pass


def generate_IC(w, ds, ix0, L, z_hat):
    '''
    Generate Initial Condition for MDE. 
    '''
    N = w.size - 1
    delta, x = make_delta(N, ix0, L)
    x0 = x[ix0]
    #u0 = delta
    u0 = np.exp(-ds*w - z_hat**2*(x[:,0]-x0)**2/(4*ds)) / np.sqrt(4*np.pi*ds)

    return z_hat*u0, x


def make_delta(N, ix0, L):
    return make_delta_heaviside(N, ix0, L)
    #return make_delta_gauss(N, ix0, L)


def make_delta_heaviside(N, ix0, L):
    '''
    Construct a delta function delta(z-z0) on Chebyshev grid.
    Approximate Dirac delta funcition by differentiating a step function.
    '''
    #ix = int(np.arccos(2*x0/L-1) / np.pi * N)
    D, x = cheb_D1_mat(N)
    H = np.zeros(N+1)
    H[0:ix0] = 1.
    H[ix0] = .5
    u = (2/L) * np.dot(D, H)

    x = .5 * (x + 1) * L

    return u, x


def make_delta_kronecker(N, x0, L):
    '''
    Construct a delta function delta(z-z0) on Chebyshev grid.
    Approximate Dirac delta funcition by Kronecker delta.
    '''
    ix = int(np.arccos(2*x0/L-1) / np.pi * N)
    u = np.zeros(N+1)
    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    w = clencurt_weights_fft(N)
    #ix = N - 1 # set the second-to-last element to be the delta postion
    u[ix] = (2.0/L) / w[ix]

    x = .5 * (x + 1) * L
    x.shape = (x.size, 1)

    return u, x, ix


def make_delta_gauss(N, ix0, L):
    '''
    Construct a delta function delta(z-z0) on Chebyshev grid.
    Approximate Dirac delta funcition by a Gaussian distribution.
    '''
    alpha = 0.001
    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    x0 = x[ix0]
    
    u = 0.5 * np.exp(-(x-x0)**2/(2*alpha)) / np.sqrt(0.5*np.pi*alpha)
    x.shape = (x.size, 1)

    return u, x


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
        self.beta = config.model.beta
        self.lbc = config.model.lbc
        self.lbc_vc = config.model.lbc_vc
        self.rbc = config.model.rbc
        self.rbc_vc = config.model.rbc_vc

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
            N = Lx - 1
            ii = np.arange(N+1)
            x = np.cos(np.pi * ii / N)
            x = .5 * (x + 1) * L
            self.w = 1 - (x/L)**2
            self.q = np.zeros((Ms, Lx))
            self.q[0, :] = 1.
            self.qc = np.zeros((Ms, Lx))
            lbc = BC(self.lbc, self.lbc_vc)
            rbc = BC(self.rbc, self.rbc_vc)
            h = 1. / (Ms - 1)
            c = 0.25 / self.beta
            self.q_solver = ETDRK4(L, N, Ms, h=h, c=c, lbc=lbc, rbc=rbc, 
                                   algo=1, scheme=1)
            #self.qc_solver = ETDRK4(L, N, Ms, h=h, c=c, lbc=lbc, rbc=rbc,
            #                        algo=1, scheme=1)
            self.qc_solver = ETDRK4(L, N, Ms-1, h=h, c=c, lbc=lbc, rbc=rbc,
                                   algo=1, scheme=1) # CKE
            #self.q_solver = OSCHEB(L, N, Ms, h)
            #self.qc_solver = OSCHEB(L, N, Ms, h)
            #self.qc_solver = OSCHEB(L, N, Ms-1, h) # CKE
        else:
            raise ValueError('Only 1D, 2D and 3D spaces are allowed!')

    def run(self):
        config = self.config
        beta = config.model.beta
        L = config.uc.a
        Lx = config.grid.Lx
        N = Lx - 1
        if self.lbc == DIRICHLET:
            x0 = 0.05 / np.sqrt(beta) # x0 = 0.1R_g = 0.1/2/beta^{1/2}
            ix0 = int(np.arccos(2*x0/L-1) / np.pi * N)
        else:
            x0 = 0.0
            ix0 = -1
        delta, x = make_delta(Lx-1, ix0, L)
        print 'x0 =', x[ix0], 'ix0 =', ix0
        Ms = config.grid.Ms
        ds = 1. / (Ms - 1)
        sigma = config.model.graft_density
        ups = config.model.excluded_volume
        lam = config.grid.lam[0]
        #x_ref = (4. * ups * sigma) ** (1./3)
        x_ref = config.model.z_hat
        phi = np.zeros(Lx)
        #phi_ref = (0.25 * sigma * sigma / ups) ** (1./3)
        phi_ref = config.model.phi_hat
        #beta = 0.25 * x_ref * x_ref
        print 'sigma =', sigma, 'ups =', ups, 'beta =', beta
        print 'Lx =', Lx, 'Ms =', Ms, 'lam =', lam
        print 'x_ref =', x_ref, 'phi_ref =', phi_ref
        display_interval = config.scft.display_interval
        record_interval = config.scft.record_interval
        save_interval = config.scft.save_interval
        save_q = config.scft.is_save_q
        thresh_residual = config.scft.thresh_residual
        data_file = os.path.join(config.scft.base_dir,
                                 config.scft.data_file)
        q_file = os.path.join(config.scft.base_dir,
                                 config.scft.q_file)
        ts = []
        Fs = []
        hs = []
        errs_residual = []
        errs_phi = []
        times = []
        t_start = clock()
        for t in xrange(1, config.scft.max_iter+1):
            #raw_input()

            # Solve MDE
            if t % display_interval == 0:
                plt.plot(x, self.w)
                plt.ylabel('w')
                plt.show()
            #    plt.plot(x_rescaled, self.q[0])
            #    plt.ylabel('q(0)')
            #    plt.show()
            self.q[0, :] = 1.
            self.q_solver.solve(self.w, self.q[0], self.q)
            if t % display_interval == 0:
                plt.plot(x, self.q[-1])
                plt.ylabel('q(-1)')
                plt.show()
            self.qc[0] = delta
            qc1, x = generate_IC(self.w, ds, ix0, L, x_ref)  # CKE
            self.qc[1] = qc1  # CKE
            if t % display_interval == 0:
                plt.plot(x, self.qc[1])
                plt.ylabel('qc[1]')
                plt.show()
            #self.qc_solver.solve(self.w, self.qc[0], self.qc)
            self.qc_solver.solve(self.w, self.qc[1], self.qc[1:])  # CKE
            if t % display_interval == 0:
                plt.plot(x, self.qc[-1])
                plt.ylabel('qc(-1)')
                plt.show()

            # Calculate density
            Q = self.q[-1, ix0]
            phi0 = phi
            phi = calc_density(self.q, self.qc) / Q
            if t % display_interval == 0:
                plt.plot(x, phi)
                plt.ylabel('$\phi$')
                plt.show()

            # Calculate energy
            F1 = -0.5 * beta * cheb_quadrature_clencurt(phi*phi)
            F2 = -np.log(Q)
            F = F1 + F2

            # Estimate error
            res = beta * phi - self.w
            if t % display_interval == 0:
                plt.plot(x, res)
                plt.ylabel('res')
                plt.show()
            err1 = np.linalg.norm(res)
            err2 = np.linalg.norm(phi-phi0)

            h = calc_brush_height(x, phi)

            if t % record_interval == 0:
                t_end = clock()
                ts.append(t)
                Fs.append(F)
                hs.append(h)
                errs_residual.append(err1)
                errs_phi.append(err2)
                times.append((t_end-t_start)/record_interval)
                print t, '\t', F1, '\t', F2, '\t', F, '\t', h
                print '\t', err1, '\t', err2
            if t % save_interval == 0:
                savemat(data_file+'_'+str(t), {'t':ts, 'time':times, 
                                    'F':Fs, 'h':hs, 'ix0':ix0,
                                    'beta':beta, 'x_ref':x_ref,
                                    'phi_ref':phi_ref, 'x':x,
                                    'err_residual':errs_residual,
                                    'err_phi':errs_phi, 
                                    'phi':phi, 'w':self.w})
                if save_q:
                    savemat(q_file+'_'+str(t), {'q':self.q, 'qc':self.qc})

            if err1 < thresh_residual:
                savemat(data_file, {'t':ts, 'time':times,
                                    'F':Fs, 'h':hs, 'ix0':ix0,
                                    'beta':beta, 'x_ref':x_ref,
                                    'phi_ref':phi_ref, 'x':x,
                                    'err_residual':errs_residual,
                                    'err_phi':errs_phi,
                                    'phi':phi, 'w':self.w})
                if save_q:
                    savemat(q_file, {'q':self.q, 'qc':self.qc})
                exit()

            # Update field
            self.w = self.w + lam * res
