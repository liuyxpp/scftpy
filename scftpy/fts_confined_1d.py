# -*- coding: utf-8 -*-
"""
fts_confined_1d
===============

FTS for confined block copolymers in 1D space.

References
----------
1. G. H. Fredrickson, "The Equilibrium Theory of Inhomogeneous Polymers", 2006, Oxford University Press: New York.
2. Alexander-Katz, A.; Moreira, A. G.; Fredrickson, G. H. J. Chem. Phys. 2003, 118, 9030–9036.

"""

import os
from time import clock
import copy

import numpy as np
from scipy.integrate import simps, trapz, romb
from scipy.io import savemat, loadmat
#from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
#from mayavi import mlab

import mpltex.acs
from chebpy import BC, ETDRK4
#from chebpy import DIRICHLET, NEUMANN, ROBIN
from chebpy import cheb_quadrature_clencurt

from scftpy import SCFTConfig, make_delta
from scftpy import quad_open4
#from scftpy import quad_semiopen4, quad_semiopen3

__all__ = ['ftsSlabAS1d', ]


def calc_density_1d(q, qc, ds):
    qqc = qc * q[::-1]
    Ms, Lx = q.shape
    phi = np.zeros(Lx)
    for i in xrange(Lx):
        phi[i] = simps(qqc[:, i], dx=ds)
        #phi[i] = quad_open4(qqc[:, i], dx=ds)

    return phi


class ftsSlabAS1d(object):
    '''
        Test: None
    '''
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
        # for AS model, config.model.f is C
        self.C = config.model.f[0]
        # for AS model, chiN is B
        self.B = config.model.chiN[0]
        self.lbc = config.model.lbc
        self.lbc_vc = config.model.lbc_vc
        self.rbc = config.model.rbc
        self.rbc_vc = config.model.rbc_vc

    def build_grid(self):
        config = self.config
        Lx = config.grid.Lx
        self.Lx = Lx
        La = config.uc.a
        self.La = La
        Ms = config.grid.Ms

        if os.path.exists(config.grid.field_data):
            mat = loadmat(config.grid.field_data)
            self.w = mat['w']
        else:
            #self.w = np.zeros([Lx])
            self.w = np.random.rand(Lx)

        self.q = np.zeros((Ms, Lx))
        self.q[0, :] = 1.

        ds = 1. / (Ms - 1)
        self.ds = ds
        lbc = BC(self.lbc, self.lbc_vc)
        rbc = BC(self.rbc, self.rbc_vc)

        self.q_solver = ETDRK4(La, Lx-1, Ms, h=ds, lbc=lbc, rbc=rbc)

        self.lam = config.grid.lam[0]

    def run(self):
        config = self.config
        C, B = self.C, self.B
        Lx, La = self.Lx, self.La
        lam = self.lam
        Ms, ds = self.config.grid.Ms, self.ds
        print 'C=', C, 'B=', B
        print 'Ms=', Ms
        print 'Lx=', Lx, 'La=', La
        print 'lam=', lam
        #config.display()

        display_interval = config.scft.display_interval
        record_interval = config.scft.record_interval
        save_interval = config.scft.save_interval
        save_q = config.scft.is_save_q
        thresh_residual = config.scft.thresh_residual
        data_file = os.path.join(config.scft.base_dir,
                                 config.scft.data_file)
        q_file = os.path.join(config.scft.base_dir,
                                 config.scft.q_file)

        phi = np.zeros([Lx])
        ii = np.arange(Lx)
        x = np.cos(np.pi * ii / (Lx - 1))
        x = 0.5 * (x + 1) * La
        ts = []
        Fs = []
        errs_residual = []
        errs_phi = []
        times = []
        t_start = clock()
        for t in xrange(1, config.scft.max_iter+1):
            # Solve MDE
            self.q_solver.solve(self.w, self.q[0], self.q)
            if t % display_interval == 0:
                plt.plot(x, self.q[-1])
                plt.xlabel('$x$')
                plt.ylabel('q[-1]')
                plt.show()

            # Calculate Q
            # Q = (1/La) * \int_0^Lx q(x, s=1) dx
            Q = 0.5 * cheb_quadrature_clencurt(self.q[-1])

            # Calculate density
            phi0 = phi
            phi = C * calc_density_1d(self.q, self.q, self.ds) / Q

            # Calculate energy
            ff = 0.5*B*C*C*phi*phi - self.w*phi
            F1 = 0.5 * cheb_quadrature_clencurt(ff)
            F2 = -C * np.log(Q)
            F = F1 + F2

            if t % display_interval == 0:
                plt.plot(x, phi, label='$\phi$')
                plt.legend(loc='best')
                plt.xlabel('$x$')
                plt.ylabel('$\phi(x)$')
                plt.show()

            res = B*C*C*phi - C * self.w
            err1 = 0.0
            err1 += np.mean(np.abs(res))

            err2 = 0.0
            err2 += np.linalg.norm(phi-phi0)

            if t % record_interval == 0 or err1 < thresh_residual:
                t_end = clock()
                ts.append(t)
                Fs.append(F)
                errs_residual.append(err1)
                errs_phi.append(err2)
                time = t_end - t_start
                times.append(time)
                print t, '\ttime =', time, '\tF =', F
                print '\tQ =', Q
                print '\t<A> =', 0.5 * cheb_quadrature_clencurt(phi),
                print '\t[', phi.min(), ', ', phi.max(), ']'
                print '\t<wA> =', 0.5 * cheb_quadrature_clencurt(self.w),
                print '\t[', self.w.min(), ', ', self.w.max(), ']'
                print '\terr1 =', err1, '\terr2 =', err2
                print
            if t % save_interval == 0:
                savemat(data_file+'_'+str(t), {'t':ts, 'time':times,
                                    'F':Fs, 'x':x,
                                    'err_residual':errs_residual,
                                    'err_phi':errs_phi,
                                    'phi':phi, 'w':self.w})
                if save_q:
                    savemat(q_file+'_'+str(t), {'q':self.q})

            if err1 < thresh_residual:
                savemat(data_file+'_'+str(t), {'t':ts, 'time':times,
                                    'F':Fs, 'x':x,
                                    'err_residual':errs_residual,
                                    'err_phi':errs_phi,
                                    'phi':phi, 'w':self.w})
                if save_q:
                    savemat(q_file+'_'+str(t), {'q':self.q})
                exit()

            # Update field
            self.w = self.w + self.lam * res
