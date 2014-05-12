# -*- coding: utf-8 -*-
"""
confined_1d
===========

SCFT for confined block copolymers in 1D space.

References
----------

"""

import os
from time import clock
import copy

import numpy as np
#from scipy.integrate import simps, trapz, romb
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

__all__ = ['SlabAB1d',
           'SlabABgC1d', ]


def calc_density_1d(q, qc, ds):
    qqc = qc * q[::-1]
    Ms, Lx = q.shape
    phi = np.zeros(Lx)
    for i in xrange(Lx):
        #phi[i] = simps(qqc[:,i], dx=ds)
        phi[i] = quad_open4(qqc[:, i], dx=ds)

    return phi


class SlabAB1d(object):
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
        self.N = config.model.N
        self.fA = config.model.f[0]
        self.a = config.model.a[0]
        self.chiN = config.model.chiN[0]
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
        MsA = config.grid.vMs[0]
        MsB = config.grid.vMs[1]
        if os.path.exists(config.grid.field_data):
            mat = loadmat(config.grid.field_data)
            self.wA = mat['wA']
            self.wB = mat['wB']
        else:
            self.wA = np.random.rand(Lx)
            self.wB = np.random.rand(Lx)
        self.qA = np.zeros((MsA, Lx))
        self.qA[0,:,:] = 1.
        self.qAc = np.zeros((MsA, Lx))
        self.qB = np.zeros((MsB, Lx))
        self.qBc = np.zeros((MsB, Lx))
        self.qBc[0,:,:] = 1.
        ds = 1. / (Ms - 1)
        self.ds = ds
        lbcA = BC(self.lbc, self.lbc_vc)
        rbcA = BC(self.rbc, self.rbc_vc)
        self.kaA = lbcA.beta; self.kbA = rbcA.beta
        fA = self.fA; fB = 1 - fA
        lbcB = copy.deepcopy(lbcA)
        lbcB.beta = -lbcA.beta * fA / fB # See Fredrickson Book p.194
        rbcB = copy.deepcopy(rbcA)
        rbcB.beta = -rbcA.beta * fA / fB
        print 'lbcA=', lbcA.__dict__
        print 'rbcA=', rbcA.__dict__
        print 'lbcB=', lbcB.__dict__
        print 'rbcB=', rbcB.__dict__
        self.qA_solver = ETDRK4(La, Lx-1, MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qAc_solver = ETDRK4(La, Lx-1, MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qB_solver = ETDRK4(La, Lx-1, MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.qBc_solver = ETDRK4(La, Lx-1, MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamY = config.grid.lam[2]
        self.yita = self.lamY # for compressible model
        #self.yita = np.zeros([Lx,Ly]) # for incompressible model

    def run(self):
        config = self.config
        print 'fA=', self.fA, 'chiN=', self.chiN
        print 'Lx=', self.Lx, 'La=', self.La
        print 'lamA=', self.lamA, 'lamB=', self.lamB, 'lamY=', self.lamY
        config.display()

        display_interval = config.scft.display_interval
        record_interval = config.scft.record_interval
        save_interval = config.scft.save_interval
        save_q = config.scft.is_save_q
        thresh_residual = config.scft.thresh_residual
        data_file = os.path.join(config.scft.base_dir,
                                 config.scft.data_file)
        q_file = os.path.join(config.scft.base_dir,
                                 config.scft.q_file)

        phiA = np.zeros([self.Lx])
        phiB = np.zeros([self.Lx])
        ii = np.arange(self.Lx)
        x = np.cos(np.pi * ii / (self.Lx - 1))
        x = 0.5 * (x + 1) * self.La
        ts = []
        Fs = []
        errs_residual = []
        errs_phi = []
        times = []
        t_start = clock()
        for t in xrange(1, config.scft.max_iter+1):
            # Solve MDE
            self.qA_solver.solve(self.wA, self.qA[0], self.qA)

            self.qB[0] = self.qA[-1]
            if t % display_interval == 0:
                plt.plot(x, self.qB[0])
                plt.xlabel('qB[0]')
                plt.show()
            self.qB_solver.solve(self.wB, self.qB[0], self.qB)
            if t % display_interval == 0:
                plt.plot(x, self.qB[-1])
                plt.xlabel('qB[-1]')
                plt.show()

            self.qBc_solver.solve(self.wB, self.qBc[0], self.qBc)

            self.qAc[0] = self.qBc[-1]
            if t % display_interval == 0:
                plt.plot(x, self.qAc[0])
                plt.xlabel('qAc[0]')
                plt.show()
            self.qAc_solver.solve(self.wA, self.qAc[0], self.qAc)
            if t % display_interval == 0:
                plt.plot(x, self.qAc[-1])
                plt.xlabel('qAc[-1]')
                plt.show()

            # Calculate Q
            # Q = (1/La/Lb) * \int_0^Lx \int_0^Ly q(x,y,s=1) dx dy
            Qx = np.mean(self.qB[-1], axis=0) # integrate along x
            Q = 0.5 * cheb_quadrature_clencurt(Qx)
            Qcx = np.mean(self.qAc[-1], axis=0) # integrate along x
            Qc = 0.5 * cheb_quadrature_clencurt(Qcx)

            # Calculate density
            phiA0 = phiA
            phiB0 = phiB
            phiA = calc_density_2d(self.qA, self.qAc, self.ds) / Q
            phiB = calc_density_2d(self.qB, self.qBc, self.ds) / Q
            #fAy = np.mean(phiA, axis=0)
            #fBy = np.mean(phiB, axis=0)
            #fAy0 = fAy[0]; fAyL = fAy[-1]
            #fBy0 = fBy[0]; fByL = fBy[-1]
            #kaB = -self.kaA * fAy0 / fBy0
            #kbB = -self.kbA * fAyL / fByL
            #self.qB_solver.lbc.beta = kaB
            #self.qB_solver.rbc.beta = kbB
            #self.qB_solver.update()
            #self.qBc_solver.lbc.beta = kaB
            #self.qBc_solver.rbc.beta = kbB
            #self.qBc_solver.update()

            # Calculate energy
            ff = self.chiN*phiA*phiB - self.wA*phiA - self.wB*phiB
            F1x = np.mean(ff, axis=0)
            F1 = 0.5 * cheb_quadrature_clencurt(F1x)
            F2 = -np.log(Q)
            F = F1 + F2

            if t % display_interval == 0:
                plt.plot(x, phiA, label='phi_A')
                plt.plot(x, phiB, label='phi_B')
                plt.legend(loc='best')
                plt.xlabel('phiA ~ x, phiB ~ x')
                plt.show()

            # Estimate error
            #resA = self.chiN*phiB - self.wA + self.yita
            #resB = self.chiN*phiA - self.wB + self.yita
            resY = phiA + phiB - 1.0
            resA = self.chiN*phiB + self.yita*resY - self.wA
            resB = self.chiN*phiA + self.yita*resY - self.wB
            err1 = 0.0
            err1 += np.mean(np.abs(resA))
            err1 += np.mean(np.abs(resB))
            #err1 += np.mean(np.abs(resY))
            err1 /= 2.
            err2 = 0.0
            err2 += np.linalg.norm(phiA-phiA0)
            err2 += np.linalg.norm(phiB-phiB0)

            if t % record_interval == 0 or err1 < thresh_residual:
                t_end = clock()
                ts.append(t)
                Fs.append(F)
                errs_residual.append(err1)
                errs_phi.append(err2)
                time = t_end - t_start
                times.append(time)
                print t, '\ttime =', time, '\tF =', F
                print '\tQ =', Q, '\tQc =', Qc
                print '\t<A> =', np.mean(phiA), '\t<B> =', np.mean(phiB)
                print '\t<wA> =', np.mean(self.wA), '\t<wB> =', np.mean(self.wB)
                #print '\tyita =', self.yita
                #print '\tkaB =', self.qB_solver.lbc.beta,
                #print '\tkbB =', self.qB_solver.rbc.beta
                print '\terr1 =', err1, '\terr2 =', err2
                print
            if t % save_interval == 0:
                savemat(data_file+'_'+str(t), {'t':ts, 'time':times,
                                    'F':Fs,
                                    'err_residual':errs_residual,
                                    'err_phi':errs_phi,
                                    'phiA':phiA, 'wA':self.wA,
                                    'phiB':phiB, 'wB':self.wB,
                                    'yita':self.yita})
                if save_q:
                    savemat(q_file+'_'+str(t), {'qA':self.qA, 'qAc':self.qAc,
                                                'qB':self.qB, 'qBc':self.qBc})

            if err1 < thresh_residual:
                savemat(data_file+'_'+str(t), {'t':ts, 'time':times,
                                    'F':Fs,
                                    'err_residual':errs_residual,
                                    'err_phi':errs_phi,
                                    'phiA':phiA, 'wA':self.wA,
                                    'phiB':phiB, 'wB':self.wB,
                                    'yita':self.yita})
                if save_q:
                    savemat(q_file+'_'+str(t), {'qA':self.qA, 'qAc':self.qAc,
                                                'qB':self.qB, 'qBc':self.qBc})
                exit()

            # Update field
            self.wA = self.wA + self.lamA * resA
            self.wB = self.wB + self.lamB * resB
            #self.yita = self.yita + self.lamY * resY


class SlabABgC1d(object):
    '''
        Model: A-B diblock + grafted C
        Confinement: parallel flat surfaces
        Test: PASSED, 2014.04.28
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
        self.fA = config.model.f[0]
        self.fB = config.model.f[1]
        self.fC = config.model.f[2]
        self.chiN = config.model.chiN[0]
        self.chiACN = config.model.chiN[1]
        self.chiBCN = config.model.chiN[2]
        self.sigma = config.model.graft_density
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
        MsA = config.grid.vMs[0]
        MsB = config.grid.vMs[1]
        MsC = config.grid.vMs[2]

        if os.path.exists(config.grid.field_data):
            mat = loadmat(config.grid.field_data)
            self.wA = mat['wA']
            self.wB = mat['wB']
            self.wC = mat['wC']
        else:
            self.wA = np.random.rand(Lx)
            self.wB = np.random.rand(Lx)
            self.wC = np.random.rand(Lx)

        self.qA = np.zeros((MsA, Lx))
        self.qA[0, :] = 1.
        self.qAc = np.zeros((MsA, Lx))
        self.qB = np.zeros((MsB, Lx))
        self.qBc = np.zeros((MsB, Lx))
        self.qBc[0, :] = 1.
        self.qC = np.zeros((MsC, Lx))
        self.qC[0, :] = 1.
        self.qCc = np.zeros((MsC, Lx))

        ds = 1. / (Ms - 1)
        self.ds = ds
        lbcA = BC(self.lbc, self.lbc_vc)
        rbcA = BC(self.rbc, self.rbc_vc)
        self.kaA = lbcA.beta
        self.kbA = rbcA.beta
        fA = self.fA
        fB = self.fB
        lbcB = copy.deepcopy(lbcA)
        lbcB.beta = -lbcA.beta * fA / fB  # See Fredrickson Book p.194
        rbcB = copy.deepcopy(rbcA)
        rbcB.beta = -rbcA.beta * fA / fB
        lbcC = copy.deepcopy(lbcA)
        rbcC = copy.deepcopy(rbcA)

        self.qA_solver = ETDRK4(La, Lx-1, MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qAc_solver = ETDRK4(La, Lx-1, MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qB_solver = ETDRK4(La, Lx-1, MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.qBc_solver = ETDRK4(La, Lx-1, MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.qC_solver = ETDRK4(La, Lx-1, MsC, h=ds, lbc=lbcC, rbc=rbcC)
        #self.qCc_solver = ETDRK4(La, Lx-1, MsC-1, h=ds, lbc=lbcC, rbc=rbcC)
        self.qCc_solver = ETDRK4(La, Lx-1, MsC, h=ds, lbc=lbcC, rbc=rbcC)

        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamC = config.grid.lam[2]
        self.lamY = config.grid.lam[3]
        #self.yita = self.lamY  # for compressible model
        self.yita = np.zeros(Lx)  # for incompressible model

    def run(self):
        config = self.config
        fA, fB, fC = self.fA, self.fB, self.fC
        chiN, chiACN, chiBCN = self.chiN, self.chiACN, self.chiBCN
        Lx, La = self.Lx, self.La
        lamA, lamB, lamC, lamY = self.lamA, self.lamB, self.lamC, self.lamY
        ds = self.ds
        sigma = self.sigma
        print 'fA=', fA, 'fB=', fB, 'fC', fC
        print 'MsA=', self.config.grid.vMs[0],
        print 'MsB=', self.config.grid.vMs[1],
        print 'MsC=', self.config.grid.vMs[2]
        print 'chiAB*N=', chiN, 'chiBC*N', chiACN, 'chiBC*N=', chiBCN
        print 'Lx=', Lx, 'La=', La
        print 'lamA=', lamA, 'lamB=', lamB, 'lamC=', lamC, 'lamY=', lamY
        #config.display()

        #if self.lbcC == DIRICHLET:
        #    ix0 = -2
        #else:
        ix0 = -1  # DBC is not supported currently
        delta, x = make_delta(Lx-1, ix0, La)
        print 'x0=', x[ix0], 'ix0=', ix0

        display_interval = config.scft.display_interval
        record_interval = config.scft.record_interval
        save_interval = config.scft.save_interval
        save_q = config.scft.is_save_q
        thresh_residual = config.scft.thresh_residual
        data_file = os.path.join(config.scft.base_dir,
                                 config.scft.data_file)
        q_file = os.path.join(config.scft.base_dir,
                              config.scft.q_file)

        phiA = np.zeros([Lx])
        phiB = np.zeros([Lx])
        phiC = np.zeros([Lx])
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
            self.qA_solver.solve(self.wA, self.qA[0], self.qA)

            self.qB[0] = self.qA[-1]
            if t % display_interval == 0:
                plt.plot(x, self.qB[0])
                plt.xlabel('qB[0]')
                plt.show()
            self.qB_solver.solve(self.wB, self.qB[0], self.qB)
            if t % display_interval == 0:
                plt.plot(x, self.qB[-1])
                plt.xlabel('qB[-1]')
                plt.show()

            self.qBc_solver.solve(self.wB, self.qBc[0], self.qBc)

            self.qAc[0] = self.qBc[-1]
            if t % display_interval == 0:
                plt.plot(x, self.qAc[0])
                plt.xlabel('qAc[0]')
                plt.show()
            self.qAc_solver.solve(self.wA, self.qAc[0], self.qAc)
            if t % display_interval == 0:
                plt.plot(x, self.qAc[-1])
                plt.xlabel('qAc[-1]')
                plt.show()

            self.qC_solver.solve(self.wC, self.qC[0], self.qC)

            self.qCc[0] = 2.0 * delta / self.qC[-1, ix0]
            #qCc1, x = generate_IC(self.wC, ds, ix0, La, 1.0)  # CKE
            #self.qCc[1] = qCc1
            if t % display_interval == 0:
                plt.plot(x, self.qCc[0])
                plt.xlabel('qCc[1]')
                plt.show()
            self.qCc_solver.solve(self.wC, self.qCc[0], self.qCc)
            #self.qCc_solver.solve(self.wC, self.qCc[1], self.qCc[1:])
            if t % display_interval == 0:
                plt.plot(x, self.qCc[-1])
                plt.xlabel('qCc[-1]')
                plt.show()

            # Calculate Q
            # Q = (1/La) * \int_0^Lx q(x,s=1) dx
            Q_AB = 0.5 * cheb_quadrature_clencurt(self.qB[-1])
            Qc_AB = 0.5 * cheb_quadrature_clencurt(self.qAc[-1])
            Q_C = self.qC[-1, ix0]

            # Calculate density
            phiA0 = phiA
            phiB0 = phiB
            phiC0 = phiC
            # Don't divide Q_AB here to intentionally force Q_AB = 1.0
            # which stablize the algorithm.
            c_AB = 1. / (1 + sigma*fC)
            phiA = c_AB * calc_density_1d(self.qA, self.qAc, ds)
            phiB = c_AB * calc_density_1d(self.qB, self.qBc, ds)
            c_C = sigma * c_AB * La
            phiC = c_C * calc_density_1d(self.qC, self.qCc, ds)

            # Calculate energy
            ff = chiN*phiA*phiB + chiACN*phiA*phiC + chiBCN*phiB*phiC
            ff = ff - self.wA*phiA - self.wB*phiB - self.wC*phiC
            F1 = 0.5 * cheb_quadrature_clencurt(ff)
            F2 = -c_AB*np.log(Q_AB) - c_C*np.log(Q_C)
            F = F1 + F2

            #if t % display_interval == 0:
            if t % 1000 == 0:
                plt.plot(x, phiA, label='$\phi_A$')
                plt.plot(x, phiB, label='$\phi_B$')
                plt.plot(x, phiC, label='$\phi_C$')
                plt.legend(loc='best')
                plt.ylabel('$\phi(x)$')
                plt.xlabel('$x$')
                plt.show()

            # Estimate error
            # incompressible model
            resA = chiN*phiB + chiACN*phiC + self.yita - self.wA
            resB = chiN*phiA + chiBCN*phiC + self.yita - self.wB
            resC = chiACN*phiA + chiBCN*phiB + self.yita - self.wC
            # compresible model
            resY = phiA + phiB + phiC - 1.0
            #resA = chiN*phiB + chiACN*phiC + self.yita*resY - self.wA
            #resB = chiN*phiA + chiBCN*phiC + self.yita*resY - self.wB
            #resC = chiACN*phiA + chiBCN*phiB + self.yita*resY - self.wC
            err1 = 0.0
            err1 += np.mean(np.abs(resA))
            err1 += np.mean(np.abs(resB))
            err1 += np.mean(np.abs(resC))
            # ONLY for incompressible model
            err1 += np.mean(np.abs(resY))
            err1 /= 4.

            err2 = 0.0
            err2 += np.linalg.norm(phiA-phiA0)
            err2 += np.linalg.norm(phiB-phiB0)
            err2 += np.linalg.norm(phiC-phiC0)
            err2 /= 3.

            if t % record_interval == 0 or err1 < thresh_residual:
                t_end = clock()
                ts.append(t)
                Fs.append(F)
                errs_residual.append(err1)
                errs_phi.append(err2)
                time = t_end - t_start
                times.append(time)
                print t, '\ttime =', time, '\tF =', F
                print '\tQ_AB =', Q_AB, '\tQc_AB =', Qc_AB,
                print '\tQ_C =', Q_C
                print '\t<A> =', 0.5 * cheb_quadrature_clencurt(phiA),
                print '\t<B> =', 0.5 * cheb_quadrature_clencurt(phiB),
                print '\t<C> =', 0.5 * cheb_quadrature_clencurt(phiC)
                print '\t<wA> =', 0.5 * cheb_quadrature_clencurt(self.wA),
                print '\t<wB> =', 0.5 * cheb_quadrature_clencurt(self.wB),
                print '\t<wC> =', 0.5 * cheb_quadrature_clencurt(self.wC)
                #print '\tyita =', self.yita
                #print '\tkaB =', self.qB_solver.lbc.beta,
                #print '\tkbB =', self.qB_solver.rbc.beta
                print '\terr1 =', err1, '\terr2 =', err2
                print
            if t % save_interval == 0:
                savemat(data_file+'_'+str(t), {'t': ts, 'time': times,
                        'F': Fs,
                        'err_residual': errs_residual,
                        'err_phi': errs_phi,
                        'phiA': phiA, 'wA': self.wA,
                        'phiB': phiB, 'wB': self.wB,
                        'phiC': phiC, 'wC': self.wC,
                        'yita': self.yita})
                if save_q:
                    savemat(q_file+'_'+str(t), {'qA': self.qA, 'qAc': self.qAc,
                                                'qB': self.qB, 'qBc': self.qBc,
                                                'qC': self.qC, 'qCc': self.qCc}
                            )

            if err1 < thresh_residual:
                savemat(data_file+'_'+str(t), {'t': ts, 'time': times,
                        'F': Fs,
                        'err_residual': errs_residual,
                        'err_phi': errs_phi,
                        'phiA': phiA, 'wA': self.wA,
                        'phiB': phiB, 'wB': self.wB,
                        'phiC': phiC, 'wC': self.wC,
                        'yita': self.yita})
                if save_q:
                    savemat(q_file+'_'+str(t), {'qA': self.qA, 'qAc': self.qAc,
                                                'qB': self.qB, 'qBc': self.qBc,
                                                'qC': self.qC, 'qCc': self.qCc}
                            )
                exit()

            # Update fields
            self.wA = self.wA + self.lamA * resA
            self.wB = self.wB + self.lamB * resB
            self.wC = self.wC + self.lamC * resC
            self.yita = self.yita + lamY * resY  # incompressible model
