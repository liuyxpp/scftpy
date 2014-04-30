# -*- coding: utf-8 -*-
"""
confined
========

SCFT for confined block copolymers.

References
----------

"""

import os
from time import clock
import copy

import numpy as np
#from scipy.integrate import simps, trapz, romb
from scipy.io import savemat, loadmat
#from scipy.fftpack import fft, ifft, fft2, ifft2, fftn, ifftn
import matplotlib.pyplot as plt
#from mayavi import mlab

import mpltex.acs
from chebpy import BC, ETDRK4FxCy, ETDRK4Polar
#from chebpy import DIRICHLET, NEUMANN, ROBIN
from chebpy import cheb_quadrature_clencurt

from scftpy import SCFTConfig, scft_contourf, make_delta
from scftpy import quad_open4
#from scftpy import quad_semiopen4, quad_semiopen3

__all__ = ['SlabAB2d',
           'SlabABgC2d',
           'DiskAB', ]


def calc_density_2d(q, qc, ds):
    qqc = qc * q[::-1]
    Ms, Lx, Ly = q.shape
    phi = np.zeros([Lx, Ly])
    for i in xrange(Lx):
        for j in xrange(Ly):
            phi[i, j] = quad_open4(qqc[:, i, j], dx=ds)
            #phi[i, j] = simps(qqc[:, i, j], dx=ds)
            #phi[i,j] = trapz(qqc[:,i,j], dx=ds)

    return phi


class SlabAB2d(object):
    '''
        Self-consistent field theory calculation of diblock copolymers confined
        in a slab in 2D (a slit). The slit is in x and its normal is in z
        direction.
        The compressible Helfand model is used, where the following energy
        penalty is added
                exp(-\beta U_c) = exp[-\yita/2 \int d^3r(\phi_A(r) + \phi_B(r)
                - 1)^2].
        Then the set of SCFT equations are
            w_A(r) = \chi N \phi_B(r) + \yita N(\phi_A(r) + \phi_B(r) - 1)
            w_B(r) = \chi N \phi_A(r) + \yita N(\phi_A(r) + \phi_B(r) - 1)
        The simple Euler relaxation scheme is used for SCFT iterations.
        The propagators are caculated by solving MDE coupled with different
        boundary conditions, such as NBC, RBC, and DBC in the z direction.
        Periodic boundary conditions are imposed on x directions.
            dqA/ds = (d^2/dx^2 + d^2/dy^2)qA - wA qA
        with boundary condtions
            [dqA/dx + kaA qA] = 0, for y=0
            [dqA/dx + kbA qA] = 0, for y=l_y
            qA(s=0) = 1, for y in (0, l_y)
        and
            dqB/ds = (d^2/dx^2 + d^2/dy^2)qB - wB qB
        with boundary condtions
            [dqB/dx + kaB qB] = 0, for y=0
            [dqB/dx + kbB qB] = 0, for y=l_x
            qB(s=0) = qA(s=1), for y in (0, l_y)
        and
            dqBc/ds = (d^2/dx^2 + d^2/dy^2)qBc - wB qBc
        with boundary condtions
            [dqBc/dx + kaB qBc] = 0, for y=0
            [dqBc/dx + kbB qBc] = 0, for y=l_y
            qBc(s=0) = 1, for y in (0, l_y)
        and
            dqAc/ds = (d^2/dx^2 + d^2/dy^2)qAc - wA qAc
        with boundary condtions
            [dqAc/dx + kaA qAc] = 0, for y=0
            [dqAc/dx + kbA qAc] = 0, for y=l_y
            qA(s=0) = qBc(s=1), for y in (0, l_y)
        ETDRK4 is employed to solve above MDEs. First these equations are
        transformed to Fourier space in the x direction
            dqk/ds = (d^2/dy - k_x^2) qk - F(wq)
        where q denotes the propagator in real space, and qk is obtained by
        conducting FFT in x direction of q, and F is the x direction Fourier
        transformation.
        In the Chebyshev-Gauss-Lobatto grid, using Chebyshev collocation
        methods, the transformed equation can be written in a set of ODEs:
            dqk/ds = [D^2 - k_x^2*I] qk - F(wq)
        where D^2 is an appropriate 2n order Chebyshev differentiation matrix
        To satisfy the incompressibility condition, we must have
            [kaA * <\phi_A(r)> + kaB * <\phi_B(r)>]_{y=0} = 0
        For weak interactions, <\phi_A(r)>|{y=0} = fA, <\phi_B(r)>|{y=0} = fB.
        So only kaA and kbA is needed, kaB and kbB can be obtained via
                kaA * fA + kaB * fB = 0
                kbA * fA + kbB * fB = 0
        (Ref: Fredrickson's book, 2006, page 194.)
        Special attentions should be paid on the y direction since the grid
        points are not equally spaced. Therefore to integrate quantities in the space, one
        should use Gauss quadrature scheme or better the Clenshaw-Curtis scheme.

        Test: PASSED 2013.08.09
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
        if os.path.exists(config.grid.field_data):
            mat = loadmat(config.grid.field_data)
            self.wA = mat['wA']
            self.wB = mat['wB']
        else:
            self.wA = np.random.rand(Lx,Ly) - 0.5
            self.wB = np.random.rand(Lx,Ly) - 0.5
        self.qA = np.zeros((MsA, Lx, Ly))
        self.qA[0,:,:] = 1.
        self.qAc = np.zeros((MsA, Lx, Ly))
        self.qB = np.zeros((MsB, Lx, Ly))
        self.qBc = np.zeros((MsB, Lx, Ly))
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
        self.qA_solver = ETDRK4FxCy(La, Lb, Lx, Ly-1,
                                             MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qAc_solver = ETDRK4FxCy(La, Lb, Lx, Ly-1,
                                              MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qB_solver = ETDRK4FxCy(La, Lb, Lx, Ly-1,
                                             MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.qBc_solver = ETDRK4FxCy(La, Lb, Lx, Ly-1,
                                              MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamY = config.grid.lam[2]
        self.yita = self.lamY # For compressible model
        #self.yita = np.zeros([Lx,Ly]) # For incompressible model

    def run(self):
        config = self.config
        print 'fA=', self.fA, 'chiN=', self.chiN
        print 'Lx=', self.Lx, 'Ly=', self.Ly
        print 'La=', self.La, 'Lb=', self.Lb
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

        phiA = np.zeros([self.Lx, self.Ly])
        phiB = np.zeros([self.Lx, self.Ly])
        ii = np.arange(self.Ly)
        y = np.cos(np.pi * ii / (self.Ly - 1))
        y = 0.5 * (y + 1) * self.Lb
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
            # Following codes are for strong surface interactions
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
                phiAB = phiA - phiB
                plt.imshow(phiAB)
                plt.show()
                plt.plot(y, phiA[self.Lx/2])
                plt.plot(y, phiB[self.Lx/2])
                plt.show()

            # Estimate error
            #resA = self.chiN*phiB - self.wA + self.yita # For incompressible model
            #resB = self.chiN*phiA - self.wB + self.yita # For incompressible model
            resY = phiA + phiB - 1.0
            resA = self.chiN*phiB + self.yita*resY - self.wA # For compressible model
            resB = self.chiN*phiA + self.yita*resY - self.wB # For compressible model
            err1 = 0.0
            err1 += np.mean(np.abs(resA))
            err1 += np.mean(np.abs(resB))
            #err1 += np.mean(np.abs(resY)) # For incompressible model
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
                phiA_meanx = np.mean(phiA, axis=0)
                phiA_mean = 0.5 * cheb_quadrature_clencurt(phiA_meanx)
                phiB_meanx = np.mean(phiB, axis=0)
                phiB_mean = 0.5 * cheb_quadrature_clencurt(phiB_meanx)
                print t, '\ttime =', time, '\tF =', F
                print '\tQ =', Q, '\tQc =', Qc
                print '\t<A> =', phiA_mean, '\t<B> =', phiB_mean
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
            #self.yita = self.yita + self.lamY * resY # incompressible model


class SlabABgC2d(object):
    '''
        Self-consistent field theory calculation of diblock copolymers confined
        in a slab in 2D (a slit). The slit is in x and its normal is in z
        direction.
        The compressible Helfand model is used, where the following energy
        penalty is added
                exp(-\beta U_c) = exp[-\yita/2 \int d^3r(\phi_A(r) + \phi_B(r)
                - 1)^2].
        Then the set of SCFT equations are
            w_A(r) = \chi N \phi_B(r) + \yita N(\phi_A(r) + \phi_B(r) - 1)
            w_B(r) = \chi N \phi_A(r) + \yita N(\phi_A(r) + \phi_B(r) - 1)
        The simple Euler relaxation scheme is used for SCFT iterations.
        The propagators are caculated by solving MDE coupled with different
        boundary conditions, such as NBC, RBC, and DBC in the z direction.
        Periodic boundary conditions are imposed on x directions.
            dqA/ds = (d^2/dx^2 + d^2/dy^2)qA - wA qA
        with boundary condtions
            [dqA/dx + kaA qA] = 0, for y=0
            [dqA/dx + kbA qA] = 0, for y=l_y
            qA(s=0) = 1, for y in (0, l_y)
        and
            dqB/ds = (d^2/dx^2 + d^2/dy^2)qB - wB qB
        with boundary condtions
            [dqB/dx + kaB qB] = 0, for y=0
            [dqB/dx + kbB qB] = 0, for y=l_x
            qB(s=0) = qA(s=1), for y in (0, l_y)
        and
            dqBc/ds = (d^2/dx^2 + d^2/dy^2)qBc - wB qBc
        with boundary condtions
            [dqBc/dx + kaB qBc] = 0, for y=0
            [dqBc/dx + kbB qBc] = 0, for y=l_y
            qBc(s=0) = 1, for y in (0, l_y)
        and
            dqAc/ds = (d^2/dx^2 + d^2/dy^2)qAc - wA qAc
        with boundary condtions
            [dqAc/dx + kaA qAc] = 0, for y=0
            [dqAc/dx + kbA qAc] = 0, for y=l_y
            qA(s=0) = qBc(s=1), for y in (0, l_y)
        ETDRK4 is employed to solve above MDEs. First these equations are
        transformed to Fourier space in the x direction
            dqk/ds = (d^2/dy - k_x^2) qk - F(wq)
        where q denotes the propagator in real space, and qk is obtained by
        conducting FFT in x direction of q, and F is the x direction Fourier
        transformation.
        In the Chebyshev-Gauss-Lobatto grid, using Chebyshev collocation
        methods, the transformed equation can be written in a set of ODEs:
            dqk/ds = [D^2 - k_x^2*I] qk - F(wq)
        where D^2 is an appropriate 2nd order Chebyshev differentiation matrix
        To satisfy the incompressibility condition, we must have
            [kaA * <\phi_A(r)> + kaB * <\phi_B(r)>]_{y=0} = 0
        For weak interactions, <\phi_A(r)>|{y=0} = fA, <\phi_B(r)>|{y=0} = fB.
        So only kaA and kbA is needed, kaB and kbB can be obtained via
                kaA * fA + kaB * fB = 0
                kbA * fA + kbB * fB = 0
        (Ref: Fredrickson's book, 2006, page 194.)
        Special attentions should be paid on the y direction since the grid
        points are not equally spaced. Therefore to integrate quantities in
        the space, one should use Gauss quadrature scheme or better the Clenshaw-Curtis scheme.

        Test: PASSED 2013.08.09
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
        if os.path.exists(config.grid.field_data):
            mat = loadmat(config.grid.field_data)
            self.wA = mat['wA']
            self.wB = mat['wB']
        else:
            self.wA = np.random.rand(Lx,Ly) - 0.5
            self.wB = np.random.rand(Lx,Ly) - 0.5
        self.qA = np.zeros((MsA, Lx, Ly))
        self.qA[0,:,:] = 1.
        self.qAc = np.zeros((MsA, Lx, Ly))
        self.qB = np.zeros((MsB, Lx, Ly))
        self.qBc = np.zeros((MsB, Lx, Ly))
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
        self.qA_solver = ETDRK4FxCy(La, Lb, Lx, Ly-1,
                                             MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qAc_solver = ETDRK4FxCy(La, Lb, Lx, Ly-1,
                                              MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qB_solver = ETDRK4FxCy(La, Lb, Lx, Ly-1,
                                             MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.qBc_solver = ETDRK4FxCy(La, Lb, Lx, Ly-1,
                                              MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamY = config.grid.lam[2]
        self.yita = self.lamY # For compressible model
        #self.yita = np.zeros([Lx,Ly]) # For incompressible model

    def run(self):
        config = self.config
        print 'fA=', self.fA, 'chiN=', self.chiN
        print 'Lx=', self.Lx, 'Ly=', self.Ly
        print 'La=', self.La, 'Lb=', self.Lb
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

        phiA = np.zeros([self.Lx, self.Ly])
        phiB = np.zeros([self.Lx, self.Ly])
        ii = np.arange(self.Ly)
        y = np.cos(np.pi * ii / (self.Ly - 1))
        y = 0.5 * (y + 1) * self.Lb
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
            # Following codes are for strong surface interactions
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
                phiAB = phiA - phiB
                plt.imshow(phiAB)
                plt.show()
                plt.plot(y, phiA[self.Lx/2])
                plt.plot(y, phiB[self.Lx/2])
                plt.show()

            # Estimate error
            #resA = self.chiN*phiB - self.wA + self.yita # For incompressible model
            #resB = self.chiN*phiA - self.wB + self.yita # For incompressible model
            resY = phiA + phiB - 1.0
            resA = self.chiN*phiB + self.yita*resY - self.wA # For compressible model
            resB = self.chiN*phiA + self.yita*resY - self.wB # For compressible model
            err1 = 0.0
            err1 += np.mean(np.abs(resA))
            err1 += np.mean(np.abs(resB))
            #err1 += np.mean(np.abs(resY)) # For incompressible model
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
                phiA_meanx = np.mean(phiA, axis=0)
                phiA_mean = 0.5 * cheb_quadrature_clencurt(phiA_meanx)
                phiB_meanx = np.mean(phiB, axis=0)
                phiB_mean = 0.5 * cheb_quadrature_clencurt(phiB_meanx)
                print t, '\ttime =', time, '\tF =', F
                print '\tQ =', Q, '\tQc =', Qc
                print '\t<A> =', phiA_mean, '\t<B> =', phiB_mean
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
            #self.yita = self.yita + self.lamY * resY # For incompressible model


class DiskAB(object):
    '''
        Self-consistent field theory calculation of diblock copolymers confined
        in a disk (2D). We consider polar coordinates.
        The compressible Helfand model is used, where the following energy
        penalty is added
                exp(-\beta U_c) = exp[-\yita/2 \int d^3r(\phi_A(r) + \phi_B(r)
                - 1)^2].
        Then the set of SCFT equations are
            w_A(r) = \chi N \phi_B(r) + \yita N(\phi_A(r) + \phi_B(r) - 1)
            w_B(r) = \chi N \phi_A(r) + \yita N(\phi_A(r) + \phi_B(r) - 1)
        The simple Euler relaxation scheme is used for SCFT iterations.
        The propagators are caculated by solving MDE coupled with different
        boundary conditions, such as NBC, RBC, and DBC in the radial direction.
        Periodic boundary conditions are imposed on \theta direction.
            dqA/ds = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dt^2)qA - wA qA
        in the domain
            [0, R] x [0, 2\pi]
        with boundary condtions
            [dqA/dr + kaA qA] = 0, for r=R
            qA(s=0) = 1, in the disk domain excluding boundary.
        and
            dqB/ds = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dt^2)qB - wB qB
        with boundary condtions
            [dqB/dr + kaB qB] = 0, for r=R
            qB(s=0) = qA(s=1), in the disk domain excluding boundary.
        and
            dqBc/ds = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dt^2)qBc - wB qBc
        with boundary condtions
            [dqBc/dr + kaB qBc] = 0, for r=R
            qBc(s=0) = 1, in the disk domain excluding boundary.
        and
            dqAc/ds = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dt^2)qAc - wA qAc
        with boundary condtions
            [dqAc/dr + kaA qAc] = 0, for r=R
            qA(s=0) = qBc(s=1), in the disk domain excluding boundary.
        ETDRK4 is employed to solve above MDEs.
        First these equations are transformed to Fourier space in the \theta direction
            dqk/ds = (d^2/dr^2 + (1/r)d/dr - (1/r^2)k_x^2) qk - F(wq)
        where q denotes the propagator in real space, and qk is obtained by
        conducting FFT in \theta direction of q, and F is the \theta direction Fourier
        transformation.
        In the Chebyshev-Gauss-Lobatto grid, using Chebyshev collocation
        methods, the transformed equation can be written in a set of ODEs:
            dqk/ds = [(D1+D2) + (M*E1+M*E2) - M**2*(k_x^2*I)] qk - F(wq)
        where D1, D2 are submatrix of the 2nd order Chebyshev differentiation
        matrix; E1, E2 are submatrix of the 1st order Chebyshev differentiation
        matrix. The details are covered in:
            Notebook page 2013.08.15 and LN Trefethen's book.

        To satisfy the incompressibility condition, we must have
            [kaA * <\phi_A> + kaB * <\phi_B>]_{r=R} = 0
        For weak interactions, <\phi_A>|{r=R} = fA, <\phi_B>|{r=R} = fB.
        So only kaA is needed, kaB can be obtained via
                kaA * fA + kaB * fB = 0
        (Ref: Fredrickson's book, 2006, page 194.)
        Special attentions should be paid on the r direction since the grid
        points are not equally spaced. Therefore to integrate quantities in the
        space, one should use Gauss quadrature scheme or better the Clenshaw-Curtis scheme.

        Test: PASSED 2013.8.15.
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
        Nt = config.grid.Lx
        Nr = config.grid.Ly # Nr is the number of grid points, EVEN only.
        Nr2 = Nr / 2
        self.Nt = Nt
        self.Nr = Nr
        self.Nr2 = Nr2
        R = config.uc.a
        self.R = R
        Ms = config.grid.Ms
        MsA = config.grid.vMs[0]
        MsB = config.grid.vMs[1]
        # Initiate fields from file or random numbers
        if os.path.exists(config.grid.field_data):
            mat = loadmat(config.grid.field_data)
            self.wA = mat['wA']
            self.wB = mat['wB']
        else:
            self.wA = np.random.rand(Nt,Nr2) - 0.5
            self.wB = np.random.rand(Nt,Nr2) - 0.5
        self.qA = np.zeros((MsA, Nt, Nr2))
        self.qA[0,:,:] = 1.
        self.qAc = np.zeros((MsA, Nt, Nr2))
        self.qB = np.zeros((MsB, Nt, Nr2))
        self.qBc = np.zeros((MsB, Nt, Nr2))
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
        self.qA_solver = ETDRK4Polar(R, Nr-1, Nt,
                                             MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qAc_solver = ETDRK4Polar(R, Nr-1, Nt,
                                              MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qB_solver = ETDRK4Polar(R, Nr-1, Nt,
                                             MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.qBc_solver = ETDRK4Polar(R, Nr-1, Nt,
                                              MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamY = config.grid.lam[2]
        self.yita = self.lamY # For compressible model
        #self.yita = np.zeros([Nt,Nr2]) # For incompressible model

    def run(self):
        config = self.config
        print 'fA=', self.fA, 'chiN=', self.chiN
        print 'Nt=', self.Nt, 'Nr=', self.Nr
        print 'R=', self.R
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

        phiA = np.zeros([self.Nt, self.Nr2])
        phiB = np.zeros([self.Nt, self.Nr2])
        tt = np.arange(self.Nt) * 2 * np.pi / self.Nt
        ii = np.arange(self.Nr)
        rr = np.cos(np.pi * ii / (self.Nr - 1))
        r2 = self.R * rr[:self.Nr2]
        rxy, txy = np.meshgrid(r2, tt)
        x, y= rxy*np.cos(txy), rxy*np.sin(txy)
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
                scft_contourf(x, y, self.qB[0])
                plt.xlabel('qB[0]')
                plt.show()
            self.qB_solver.solve(self.wB, self.qB[0], self.qB)
            if t % display_interval == 0:
                scft_contourf(x, y, self.qB[-1])
                plt.xlabel('qB[-1]')
                plt.show()

            self.qBc_solver.solve(self.wB, self.qBc[0], self.qBc)

            self.qAc[0] = self.qBc[-1]
            if t % display_interval == 0:
                scft_contourf(x, y, self.qAc[0])
                plt.xlabel('qAc[0]')
                plt.show()
            self.qAc_solver.solve(self.wA, self.qAc[0], self.qAc)
            if t % display_interval == 0:
                scft_contourf(x, y, self.qAc[-1])
                plt.xlabel('qAc[-1]')
                plt.show()

            # Calculate Q
            # Q = (1/V) * \int_0^R \int_0^{2\pi} q(x,y,s=1) r dr d\theta
            # V = \pi R^2
            #print np.max(self.qB[-1]), np.min(self.qB[-1])
            Qt = np.mean(self.qB[-1], axis=0) # integrate along \theta
            Qt = np.hstack((Qt, -Qt[::-1]))
            Q = cheb_quadrature_clencurt(rr*Qt)
            Qct = np.mean(self.qAc[-1], axis=0) # integrate along \theta
            Qct = np.hstack((Qct, -Qct[::-1]))
            Qc = cheb_quadrature_clencurt(rr*Qct)

            # Calculate density
            phiA0 = phiA
            phiB0 = phiB
            phiA = calc_density_2d(self.qA, self.qAc, self.ds) / Q
            phiB = calc_density_2d(self.qB, self.qBc, self.ds) / Q
            # Following codes are for strong surface interactions
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
            F1t = np.mean(ff, axis=0)
            F1t = np.hstack((F1t,-F1t[::-1]))
            F1 = cheb_quadrature_clencurt(rr*F1t)
            F2 = -np.log(Q)
            F = F1 + F2

            if t % display_interval == 0:
                phiAB = phiA - phiB
                scft_contourf(x, y, phiAB)
                plt.xlabel('phiA-phiB')
                plt.show()
                plt.plot(r2, phiA[self.Nt/2])
                plt.plot(r2, phiB[self.Nt/2])
                plt.show()

            # Estimate error
            #resA = self.chiN*phiB - self.wA + self.yita # For incompressible model
            #resB = self.chiN*phiA - self.wB + self.yita # For incompressible model
            resY = phiA + phiB - 1.0
            resA = self.chiN*phiB + self.yita*resY - self.wA # For compressible model
            resB = self.chiN*phiA + self.yita*resY - self.wB # For compressible model
            err1 = 0.0
            err1 += np.mean(np.abs(resA))
            err1 += np.mean(np.abs(resB))
            #err1 += np.mean(np.abs(resY)) # For incompressible model
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
                phiA_meanx = np.mean(phiA, axis=0)
                phiA_meanx = np.hstack((phiA_meanx, -phiA_meanx[::-1]))
                phiA_mean = cheb_quadrature_clencurt(rr*phiA_meanx)
                phiB_meanx = np.mean(phiB, axis=0)
                phiB_meanx = np.hstack((phiB_meanx, -phiB_meanx[::-1]))
                phiB_mean = cheb_quadrature_clencurt(rr*phiB_meanx)
                print t, '\ttime =', time, '\tF =', F
                print '\tQ =', Q, '\tQc =', Qc
                print '\t<A> =', phiA_mean, '\t<B> =', phiB_mean
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
            #self.yita = self.yita + self.lamY * resY # For incompressible model
