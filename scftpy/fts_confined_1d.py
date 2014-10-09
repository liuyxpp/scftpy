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

__all__ = ['ftsSlabAS1d',
           'ftsSlabAS1d_phi',
          ]


def calc_density_1d(q, qc, ds):
    qqc = qc * q[::-1]
    Ms, Lx = q.shape
    phi = np.zeros(Lx, dtype=np.complex128)
    for i in xrange(Lx):
        phi[i] = simps(qqc[:, i], dx=ds)
        #phi[i] = quad_open4(qqc[:, i], dx=ds)

    return phi


class ftsSlabAS1d(object):
    '''
        For SCFT, \phi_mean = 1.0
        For FTS, \phi_mean = C
        Test: PASSED for complex SCFT, 2014.10.07.
        Test: PASSED for FTS using Complex Langevin (CL), 2014.10.08.
        Test: None for \phi-fixed FTS using CL, 2014.10.##.
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
            self.w = mat['w'].reshape(Lx)
        else:
            self.w = np.zeros(Lx, dtype=np.complex128)
            #self.w.real = np.random.rand(Lx)
            #self.w.imag = np.random.rand(Lx)
        self.phi = np.zeros(Lx, dtype=np.complex128)

        self.q = np.zeros((Ms, Lx), dtype=np.complex128)
        self.q[0].real = 1.

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
        dx = La / Lx
        lam = self.lam
        Ms, ds = self.config.grid.Ms, self.ds
        num_eq_step = 50000
        num_avg_step = 50000
        print 'ftsSlabAS1d'
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

        phi_op = np.zeros(Lx, dtype=np.complex128)
        ii = np.arange(Lx)
        x = np.cos(np.pi * ii / (Lx - 1))
        x = 0.5 * (x + 1) * La
        ts = []
        Fs = []
        errs_residual = []
        errs_phi = []
        times = []
        mu = []
        mu_sum = 0.0
        mu_avg = 0.0
        t_avg = 0
        phi_sum = np.zeros(Lx)
        phi_avg = np.zeros(Lx)
        t_start = clock()
        for t in xrange(1, config.scft.max_iter+1):
            # Solve MDE
            self.q_solver.solve(self.w, self.q[0], self.q)
            if t % display_interval == 0:
                plt.plot(x, self.q[-1].real)
                plt.xlabel('$x$')
                plt.ylabel('q[-1]')
                plt.show()

            # Calculate Q
            # Q = (1/La) * \int_0^Lx q(x, s=1) dx
            Q = 0.5 * cheb_quadrature_clencurt(self.q[-1])

            # Calculate density operator
            #phi0 = self.phi
            phi_op = calc_density_1d(self.q, self.q, self.ds) / Q

            # Calculate energy
            ff = 0.5*B*C*C*self.phi*self.phi - C*self.phi*self.w
            F1 = 0.5 * cheb_quadrature_clencurt(ff)
            F2 = -C * np.log(Q)
            F = F1 + F2

            # Calculate chemical potential after
            if t > num_eq_step:
                t_avg += 1
                phi_sum += phi_op
                phi_avg = phi_sum / t_avg
                mu_op = -np.log(Q) + np.log(C)
                mu.append(mu_op)
                mu_sum += mu_op
                mu_avg = mu_sum / t_avg

            if t % display_interval == 0:
                if t > num_eq_step:
                    plt.plot(x, phi_avg.real, label='$<\phi>$')
                else:
                    plt.plot(x, phi_op.real, label='$\phi$')
                plt.legend(loc='best')
                plt.xlabel('$x$')
                plt.ylabel('$\phi(x)$')
                plt.show()

            force_phi = C * (B*C*self.phi - self.w)
            force_w = C * (self.phi - phi_op)
            #err1 = 0.0
            #err1 += np.mean(np.abs(force_phi.real))
            #err1 += np.mean(np.abs(force_w.real))
            #err1 /= 2

            #err2 = 0.0
            #err2 += np.linalg.norm(phi-phi0)

            if t % record_interval == 0 or t == num_avg_step+num_eq_step:
                t_end = clock()
                ts.append(t)
                Fs.append(F)
                #errs_residual.append(err1)
                #errs_phi.append(err2)
                time = t_end - t_start
                times.append(time)
                print t, '\ttime =', time, '\tF =', F
                print '\tQ =', Q, '\tmu =', mu_avg
                print '\t<A> =', 0.5 * cheb_quadrature_clencurt(phi_op),
                print '\t[', phi_op.real.min(), ', ',
                print phi_op.real.max(), ']'
                print '\t<wA> =', 0.5 * cheb_quadrature_clencurt(self.w),
                print '\t[', self.w.real.min(), ', ', self.w.real.max(), ']'
                #print '\terr1 =', err1, '\terr2 =', err2
                print
            if t % save_interval == 0:
                savemat(data_file+'_'+str(t), {'t':ts, 'time':times,
                                    'F':Fs, 'x':x,
                                    #'err_residual':errs_residual,
                                    #'err_phi':errs_phi,
                                    'mu_avg':mu_avg, 'mu':mu,
                                    'phi_avg':phi_avg, 'w':self.w})
                if save_q:
                    savemat(q_file+'_'+str(t), {'q':self.q})

            if t == num_avg_step+num_eq_step:
                savemat(data_file+'_'+str(t), {'t':ts, 'time':times,
                                    'F':Fs, 'x':x,
                                    #'err_residual':errs_residual,
                                    #'err_phi':errs_phi,
                                    'mu_avg':mu_avg, 'mu':mu,
                                    'phi_avg':phi_avg, 'w':self.w})
                if save_q:
                    savemat(q_file+'_'+str(t), {'q':self.q})
                exit()

            # Update field
            self.phi -= lam * force_phi
            self.w.real -= lam * force_w.real
            self.w.imag -= lam * force_w.imag
            self.w.imag += np.sqrt(2*lam/dx) * np.random.randn(Lx)


class ftsSlabAS1d_phi(object):
    '''
        For SCFT, \phi_mean = 1.0
        For FTS, \phi_mean = C
        Test: PASSED for complex SCFT, 2014.10.07.
        Test: PASSED for FTS using Complex Langevin (CL), 2014.10.08.
        Test: None for \phi-fixed FTS using CL, 2014.10.##.
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
            print "Reading from ", config.grid.field_data
            mat = loadmat(config.grid.field_data)
            self.w = np.zeros(Lx, dtype=np.complex128)
            self.w = mat['iw_avg'].reshape(Lx)
            self.phi = np.zeros(Lx, dtype=np.complex128)
            self.phi.real = mat['phi_avg'].reshape(Lx).real
        else:
            self.w = np.zeros(Lx, dtype=np.complex128)
            #self.w.real = np.random.rand(Lx)
            #self.w.imag = np.random.rand(Lx)
            self.phi = np.zeros(Lx, dtype=np.complex128)

        self.q = np.zeros((Ms, Lx), dtype=np.complex128)
        self.q[0].real = 1.

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
        dx = La / Lx
        lam = self.lam
        Ms, ds = self.config.grid.Ms, self.ds
        num_eq_step = 50000
        num_avg_step = 50000
        print 'ftsSlabAS1d'
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

        phi_op = np.zeros(Lx, dtype=np.complex128)
        ii = np.arange(Lx)
        x = np.cos(np.pi * ii / (Lx - 1))
        x = 0.5 * (x + 1) * La
        ts = []
        Fs = []
        errs_residual = []
        errs_phi = []
        times = []
        mu = []
        mu_sum = 0j
        mu_avg = 0j
        t_avg = 0
        phi_sum = np.zeros(Lx, dtype=np.complex128)
        phi_avg = np.zeros(Lx, dtype=np.complex128)
        iw_sum = np.zeros(Lx, dtype=np.complex128)
        iw_avg = np.zeros(Lx, dtype=np.complex128)
        t_start = clock()
        for t in xrange(1, config.scft.max_iter+1):
            # Solve MDE
            self.q_solver.solve(self.w, self.q[0], self.q)
            if t % display_interval == 0:
                plt.plot(x, self.q[-1].real)
                plt.xlabel('$x$')
                plt.ylabel('q[-1]')
                plt.show()

            # Calculate Q
            # Q = (1/La) * \int_0^Lx q(x, s=1) dx
            Q = 0.5 * cheb_quadrature_clencurt(self.q[-1])

            # Calculate density operator
            #phi0 = self.phi
            phi_op = calc_density_1d(self.q, self.q, self.ds) / Q

            # Calculate energy
            ff = 0.5*B*C*C*self.phi*self.phi - C*self.phi*self.w
            F1 = 0.5 * cheb_quadrature_clencurt(ff)
            F2 = -C * np.log(Q)
            F = F1 + F2

            # Calculate chemical potential in production runs
            if t > num_eq_step:
                t_avg += 1
                iw_sum += self.w
                iw_avg = iw_sum / t_avg
                phi_sum += phi_op
                phi_avg = phi_sum / t_avg
                mu_op = -np.log(Q) + np.log(C)
                mu.append(mu_op)
                mu_sum += mu_op
                mu_avg = mu_sum / t_avg

            if t % display_interval == 0:
                if t > num_eq_step:
                    plt.plot(x, phi_avg.real, label='$<\phi>$')
                else:
                    plt.plot(x, phi_op.real, label='$\phi$')
                plt.legend(loc='best')
                plt.xlabel('$x$')
                plt.ylabel('$\phi(x)$')
                plt.show()

            force_phi = C * (B*C*self.phi - self.w)
            force_w = C * (self.phi - phi_op)
            #err1 = 0.0
            #err1 += np.mean(np.abs(force_phi.real))
            #err1 += np.mean(np.abs(force_w.real))
            #err1 /= 2

            #err2 = 0.0
            #err2 += np.linalg.norm(phi-phi0)

            if t % record_interval == 0 or t == num_avg_step+num_eq_step:
                t_end = clock()
                ts.append(t)
                Fs.append(F)
                #errs_residual.append(err1)
                #errs_phi.append(err2)
                time = t_end - t_start
                times.append(time)
                print t, '\ttime =', time, '\tF =', F
                print '\tQ =', Q, '\tmu =', mu_avg
                print '\t<A> =', 0.5 * cheb_quadrature_clencurt(phi_op),
                print '\t[', phi_op.real.min(), ', ', phi_op.real.max(), ']'
                print '\t<wA> =', 0.5 * cheb_quadrature_clencurt(self.w),
                print '\t[', self.w.real.min(), ', ', self.w.real.max(), ']'
                #print '\terr1 =', err1, '\terr2 =', err2
                print
            if t % save_interval == 0:
                savemat(data_file+'_'+str(t), {'t':ts, 'time':times,
                                    'F':Fs, 'x':x,
                                    #'err_residual':errs_residual,
                                    #'err_phi':errs_phi,
                                    'iw_avg':iw_avg,
                                    'mu_avg':mu_avg, 'mu':mu,
                                    'phi_avg':phi_avg,
                                    'phi':self.phi, 'w':self.w})
                if save_q:
                    savemat(q_file+'_'+str(t), {'q':self.q})

            if t == num_avg_step+num_eq_step:
                savemat(data_file+'_'+str(t), {'t':ts, 'time':times,
                                    'F':Fs, 'x':x,
                                    #'err_residual':errs_residual,
                                    #'err_phi':errs_phi,
                                    'iw_avg':iw_avg,
                                    'mu_avg':mu_avg, 'mu':mu,
                                    'phi_avg':phi_avg,
                                    'phi': self.phi, 'w':self.w})
                if save_q:
                    savemat(q_file+'_'+str(t), {'q':self.q})
                exit()

            # Update field
            self.phi -= lam * force_phi
            self.w.real -= lam * force_w.real
            self.w.imag -= lam * force_w.imag
            self.w.imag += np.sqrt(2*lam/dx) * np.random.randn(Lx)
