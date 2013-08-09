# -*- coding: utf-8 -*-
"""
bulk
====

SCFT for block copolymers in bulk.

References
----------

Liu, Y. X. *Polymer Brush III*, Technical Reports, Fudan University, 2012. 

"""

import os
from time import clock

import numpy as np
from scipy.integrate import simps, trapz, romb
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
#from mayavi import mlab

import mpltex.acs
from chebpy import OSF, OSF2d, OSF3d

from scftpy import SCFTConfig

__all__ = ['BulkAB1d',
           'BulkAB2d',
           'BulkAB3d',
          ]

def calc_density_1d(q, qc, ds):
    qqc = qc * q[::-1]
    Ms, Lx = q.shape
    phi = np.zeros(Lx)
    for i in xrange(Lx):
        #phi[i] = simps(qqc[:,i], dx=ds)
        phi[i] = trapz(qqc[:,i], dx=ds)

    return phi


def calc_density_2d(q, qc, ds):
    qqc = qc * q[::-1]
    Ms, Lx, Ly = q.shape
    phi = np.zeros([Lx, Ly])
    for i in xrange(Lx):
        for j in xrange(Ly):
            #phi[i,j] = simps(qqc[:,i,j], dx=ds)
            phi[i,j] = trapz(qqc[:,i,j], dx=ds)

    return phi


def calc_density_3d(q, qc, ds):
    qqc = qc * q[::-1]
    Ms, Lx, Ly, Lz = q.shape
    phi = np.zeros([Lx, Ly, Lz])
    for i in xrange(Lx):
        for j in xrange(Ly):
            for k in xrange(Lz):
                #phi[i,j,k] = simps(qqc[:,i,j,k], dx=ds)
                phi[i,j,k] = trapz(qqc[:,i,j,k], dx=ds)

    return phi


class BulkAB1d(object):
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

    def build_grid(self):
        config = self.config
        Lx = config.grid.Lx
        self.Lx = Lx
        L = config.uc.a
        self.L = L
        Ms = config.grid.Ms
        MsA = config.grid.vMs[0]
        MsB = config.grid.vMs[1]
        self.wA = np.random.rand(Lx) - 0.5
        self.wB = np.random.rand(Lx) - 0.5
        self.yita = np.zeros(Lx)
        self.qA = np.zeros((MsA, Lx))
        self.qA[0,:] = 1.
        self.qAc = np.zeros((MsA, Lx))
        self.qB = np.zeros((MsB, Lx))
        self.qBc = np.zeros((MsB, Lx))
        self.qBc[0,:] = 1.
        ds = 1. / (Ms - 1)
        self.ds = ds
        self.qA_solver = OSF(L, Lx, MsA, ds)
        self.qAc_solver = OSF(L, Lx, MsA, ds)
        self.qB_solver = OSF(L, Lx, MsB, ds)
        self.qBc_solver = OSF(L, Lx, MsB, ds)
        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamY = config.grid.lam[2]

    def run(self):
        config = self.config
        print 'fA=', self.fA, 'chiN=', self.chiN
        print 'Lx=', self.Lx, 'L=', self.L
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

        x = np.arange(self.Lx) * self.L / self.Lx
        phiA = np.zeros(self.Lx)
        phiB = np.zeros(self.Lx)
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
                plt.ylabel('qB(0)')
                plt.show()
            self.qB_solver.solve(self.wB, self.qB[0], self.qB)
            if t % display_interval == 0:
                plt.plot(x, self.qB[-1])
                plt.ylabel('qB(-1)')
                plt.show()

            self.qBc_solver.solve(self.wB, self.qBc[0], self.qBc)

            self.qAc[0] = self.qBc[-1]
            if t % display_interval == 0:
                plt.plot(x, self.qAc[0])
                plt.ylabel('qAc(0)')
                plt.show()
            self.qAc_solver.solve(self.wA, self.qAc[0], self.qAc)
            if t % display_interval == 0:
                plt.plot(x, self.qAc[-1])
                plt.ylabel('qAc(-1)')
                plt.show()

            # Calculate density
            phiA0 = phiA
            phiB0 = phiB
            phiA = calc_density_1d(self.qA, self.qAc, self.ds)
            phiB = calc_density_1d(self.qB, self.qBc, self.ds)
            if t % display_interval == 0:
                plt.plot(x, phiA)
                plt.plot(x, phiB)
                plt.ylabel('$\phi$')
                plt.show()

            # Calculate energy
            Q = np.mean(self.qB[-1])
            F1 = np.mean(self.chiN*phiA*phiB - self.wA*phiA - self.wB*phiB)
            F2 = -np.log(Q)
            F = F1 + F2

            # Estimate error
            resA = self.chiN*phiB + self.yita - self.wA
            resB = self.chiN*phiA + self.yita - self.wB
            resY = phiA + phiB - 1.0
            err1 = 0.0
            err1 += np.mean(np.abs(resA))
            err1 += np.mean(np.abs(resB))
            err1 += np.mean(np.abs(resY))
            err2 = 0.0
            err2 += np.linalg.norm(phiA-phiA0)
            err2 += np.linalg.norm(phiB-phiB0)

            if t % record_interval == 0:
                t_end = clock()
                ts.append(t)
                Fs.append(F)
                errs_residual.append(err1)
                errs_phi.append(err2)
                times.append((t_end-t_start)/record_interval)
                print t, '\tF =', F, '\tQ =', Q
                print '\t<A> =', np.mean(phiA), '\t<B> =', np.mean(phiB)
                print '\t<wA> =', np.mean(self.wA), '\t<wB> =', np.mean(self.wB),
                print '\t<yita> =', np.mean(self.yita)
                print '\terr1 =', err1, '\terr2 =', err2
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
            self.yita = self.yita + self.lamY * resY

            
class BulkAB2d(object):
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

    def build_grid(self):
        config = self.config
        Lx = config.grid.Lx
        Ly = config.grid.Ly
        self.Lx = Lx
        self.Ly = Ly
        La = config.uc.a
        Lb = config.uc.b
        self.La = La
        self.Lb = Lb
        Ms = config.grid.Ms
        MsA = config.grid.vMs[0]
        MsB = config.grid.vMs[1]
        self.wA = np.random.rand(Lx,Ly) - 0.5
        self.wB = np.random.rand(Lx,Ly) - 0.5
        self.yita = np.zeros([Lx,Ly])
        self.qA = np.zeros((MsA, Lx, Ly))
        self.qA[0,:] = 1.
        self.qAc = np.zeros((MsA, Lx, Ly))
        self.qB = np.zeros((MsB, Lx, Ly))
        self.qBc = np.zeros((MsB, Lx, Ly))
        self.qBc[0,:] = 1.
        ds = 1. / (Ms - 1)
        self.ds = ds
        self.qA_solver = OSF2d(La, Lb, Lx, Ly, MsA, ds)
        self.qAc_solver = OSF2d(La, Lb, Lx, Ly, MsA, ds)
        self.qB_solver = OSF2d(La, Lb, Lx, Ly, MsB, ds)
        self.qBc_solver = OSF2d(La, Lb, Lx, Ly, MsB, ds)
        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamY = config.grid.lam[2]

    def run(self):
        config = self.config
        print 'fA=', self.fA, 'chiN=', self.chiN
        print 'Lx=', self.Lx, 'Ly=', self.Ly
        print 'La=', self.La, 'Ly=', self.La
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

        #x = np.arange(self.Lx) * self.La / self.Lx
        #y = np.arange(self.Ly) * self.Lb / self.Ly
        phiA = np.zeros([self.Lx, self.Ly])
        phiB = np.zeros([self.Lx, self.Ly])
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
                plt.imshow(self.qB[0])
                plt.show()
            self.qB_solver.solve(self.wB, self.qB[0], self.qB)
            if t % display_interval == 0:
                plt.imshow(self.qB[-1])
                plt.show()

            self.qBc_solver.solve(self.wB, self.qBc[0], self.qBc)

            self.qAc[0] = self.qBc[-1]
            if t % display_interval == 0:
                plt.imshow(self.qAc[0])
                plt.show()
            self.qAc_solver.solve(self.wA, self.qAc[0], self.qAc)
            if t % display_interval == 0:
                plt.imshow(self.qAc[-1])
                plt.show()

            # Calculate density
            phiA0 = phiA
            phiB0 = phiB
            phiA = calc_density_2d(self.qA, self.qAc, self.ds)
            phiB = calc_density_2d(self.qB, self.qBc, self.ds)
            if t % display_interval == 0:
                plt.imshow(phiA-phiB)
                plt.ylabel('$\phi_A-\phi_B$')
                plt.show()

            # Calculate energy
            Q = np.mean(self.qB[-1])
            F1 = np.mean(self.chiN*phiA*phiB - self.wA*phiA - self.wB*phiB)
            F2 = -np.log(Q)
            F = F1 + F2

            # Estimate error
            resA = self.chiN*phiB + self.yita - self.wA
            resB = self.chiN*phiA + self.yita - self.wB
            resY = phiA + phiB - 1.0
            err1 = 0.0
            err1 += np.mean(np.abs(resA))
            err1 += np.mean(np.abs(resB))
            err1 += np.mean(np.abs(resY))
            err2 = 0.0
            err2 += np.linalg.norm(phiA-phiA0)
            err2 += np.linalg.norm(phiB-phiB0)

            if t % record_interval == 0:
                t_end = clock()
                ts.append(t)
                Fs.append(F)
                errs_residual.append(err1)
                errs_phi.append(err2)
                times.append((t_end-t_start)/record_interval)
                print t, '\tF =', F, '\tQ =', Q
                print '\t<A> =', np.mean(phiA), '\t<B> =', np.mean(phiB)
                print '\t<wA> =', np.mean(self.wA), '\t<wB> =', np.mean(self.wB),
                print '\t<yita> =', np.mean(self.yita)
                print '\terr1 =', err1, '\terr2 =', err2
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
            self.yita = self.yita + self.lamY * resY


class BulkAB3d(object):
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

    def build_grid(self):
        config = self.config
        Lx = config.grid.Lx
        Ly = config.grid.Ly
        Lz = config.grid.Lz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        La = config.uc.a
        Lb = config.uc.b
        Lc = config.uc.c
        self.La = La
        self.Lb = Lb
        self.Lc = Lc
        Ms = config.grid.Ms
        MsA = config.grid.vMs[0]
        MsB = config.grid.vMs[1]
        self.wA = np.random.rand(Lx,Ly,Lz) - 0.5
        self.wB = np.random.rand(Lx,Ly,Lz) - 0.5
        self.yita = np.zeros([Lx,Ly,Lz])
        self.qA = np.zeros((MsA, Lx, Ly, Lz))
        self.qA[0,:] = 1.
        self.qAc = np.zeros((MsA, Lx, Ly, Lz))
        self.qB = np.zeros((MsB, Lx, Ly, Lz))
        self.qBc = np.zeros((MsB, Lx, Ly, Lz))
        self.qBc[0,:] = 1.
        ds = 1. / (Ms - 1)
        self.ds = ds
        self.qA_solver = OSF3d(La, Lb, Lc, Lx, Ly, Lz, MsA, ds)
        self.qAc_solver = OSF3d(La, Lb, Lc, Lx, Ly, Lz, MsA, ds)
        self.qB_solver = OSF3d(La, Lb, Lc, Lx, Ly, Lz, MsB, ds)
        self.qBc_solver = OSF3d(La, Lb, Lc, Lx, Ly, Lz, MsB, ds)
        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamY = config.grid.lam[2]

    def run(self):
        config = self.config
        print 'fA=', self.fA, 'chiN=', self.chiN
        print 'Lx=', self.Lx, 'Ly=', self.Ly, 'Lz=', self.Lz
        print 'La=', self.La, 'Lb=', self.Lb, 'Lc=', self.Lc
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

        phiA = np.zeros([self.Lx, self.Ly, self.Lz])
        phiB = np.zeros([self.Lx, self.Ly, self.Lz])
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
                plt.imshow(self.qB[0,self.Lx/2])
                plt.show()
            self.qB_solver.solve(self.wB, self.qB[0], self.qB)
            if t % display_interval == 0:
                plt.imshow(self.qB[-1,self.Lx/2])
                plt.show()

            self.qBc_solver.solve(self.wB, self.qBc[0], self.qBc)

            self.qAc[0] = self.qBc[-1]
            if t % display_interval == 0:
                plt.imshow(self.qAc[0,self.Lx/2])
                plt.show()
            self.qAc_solver.solve(self.wA, self.qAc[0], self.qAc)
            if t % display_interval == 0:
                plt.imshow(self.qAc[-1,self.Lx/2])
                plt.show()

            # Calculate density
            phiA0 = phiA
            phiB0 = phiB
            phiA = calc_density_3d(self.qA, self.qAc, self.ds)
            phiB = calc_density_3d(self.qB, self.qBc, self.ds)
            if t % display_interval == 0:
                phiAB = phiA - phiB
                plt.imshow(phiAB[self.Lx/2])
                plt.show()

            # Calculate energy
            Q = np.mean(self.qB[-1])
            F1 = np.mean(self.chiN*phiA*phiB - self.wA*phiA - self.wB*phiB)
            F2 = -np.log(Q)
            F = F1 + F2

            # Estimate error
            resA = self.chiN*phiB + self.yita - self.wA
            resB = self.chiN*phiA + self.yita - self.wB
            resY = phiA + phiB - 1.0
            err1 = 0.0
            err1 += np.mean(np.abs(resA))
            err1 += np.mean(np.abs(resB))
            err1 += np.mean(np.abs(resY))
            err2 = 0.0
            err2 += np.linalg.norm(phiA-phiA0)
            err2 += np.linalg.norm(phiB-phiB0)

            if t % record_interval == 0:
                t_end = clock()
                ts.append(t)
                Fs.append(F)
                errs_residual.append(err1)
                errs_phi.append(err2)
                times.append((t_end-t_start)/record_interval)
                print t, '\tF =', F, '\tQ =', Q
                print '\t<A> =', np.mean(phiA), '\t<B> =', np.mean(phiB)
                print '\t<wA> =', np.mean(self.wA), '\t<wB> =', np.mean(self.wB),
                print '\t<yita> =', np.mean(self.yita)
                print '\terr1 =', err1, '\terr2 =', err2
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
            self.yita = self.yita + self.lamY * resY


