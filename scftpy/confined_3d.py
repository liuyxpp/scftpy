# -*- coding: utf-8 -*-
"""
confined_3d
===========

SCFT for confined block copolymers in 3D space.

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
from chebpy import BC, ETDRK4FxyCz, ETDRK4Cylind
#from chebpy import DIRICHLET, NEUMANN, ROBIN
from chebpy import cheb_quadrature_clencurt

from scftpy import SCFTConfig, contourf_slab2d, make_delta
from scftpy import quad_open4
#from scftpy import quad_semiopen4, quad_semiopen3

__all__ = ['SlabAB3d',
           'SlabABgC3d',
           'CylinderAB', ]


def calc_density_3d(q, qc, ds):
    qqc = qc * q[::-1]
    Ms, Lx, Ly, Lz = q.shape
    phi = np.zeros([Lx, Ly, Lz])
    for i in xrange(Lx):
        for j in xrange(Ly):
            for k in xrange(Lz):
                phi[i, j, k] = quad_open4(qqc[:, i, j, k], dx=ds)
                #phi[i,j,k] = simps(qqc[:,i,j,k], dx=ds)
                # phi[i, j, k] = trapz(qqc[:, i, j, k], dx=ds)

    return phi


class SlabAB3d(object):
    '''
        Self-consistent field theory calculation of diblock copolymers confined
        in a slab in 3D. The slab is in x-y plane and its normal is in z
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
        Periodic boundary conditions are imposed on x and y directions.
            dqA/ds = (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)qA - wA qA
        with boundary condtions
            [dqA/dx + kaA qA] = 0, for z=0
            [dqA/dx + kbA qA] = 0, for z=l_z
            qA(s=0) = 1, for z in (0, l_z)
        and
            dqB/ds = (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)qB - wB qB
        with boundary condtions
            [dqB/dx + kaB qB] = 0, for z=0
            [dqB/dx + kbB qB] = 0, for z=l_z
            qB(s=0) = qA(s=1), for z in (0, l_z)
        and
            dqBc/ds = (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)qBc - wB qBc
        with boundary condtions
            [dqBc/dx + kaB qBc] = 0, for z=0
            [dqBc/dx + kbB qBc] = 0, for z=l_z
            qBc(s=0) = 1, for z in (0, l_z)
        and
            dqAc/ds = (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)qAc - wA qAc
        with boundary condtions
            [dqAc/dx + kaA qAc] = 0, for z=0
            [dqAc/dx + kbA qAc] = 0, for z=l_x
            qA(s=0) = qBc(s=1), for z in (0, l_z)
        ETDRK4 is employed to solve above MDEs. First these equations are
        transformed to Fourier space in the x direction
            dqk/ds = [d^2/dy - (k_x^2 + k_y^2)] qk - F(wq)
        where q denotes the propagator in real space, and qk is obtained by
        conducting FFT in x direction of q, and F is the x direction Fourier
        transformation.
        In the Chebyshev-Gauss-Lobatto grid, using Chebyshev collocation
        methods, the transformed equation can be written in a set of ODEs:
            dqk/ds = (D^2 - (k_x^2 + k_y^2)*I) qk - F(wq)
        where D^2 is an appropriate 2n order Chebyshev differentiation matrix
        To satisfy the incompressibility condition, we must have
            [kaA * <\phi_A(r)> + kaB * <\phi_B(r)>]_{z=0} = 0
        For weak interactions, <\phi_A(r)>|{z=0} = fA, <\phi_B(r)>|{z=0} = fB.
        So only kaA and kbA is needed, kaB and kbB can be obtained via
                kaA * fA + kaB * fB = 0
                kbA * fA + kbB * fB = 0
        (Ref: Fredrickson's book, 2006, page 194.)
        Special attentions should be paid on the z direction since the grid
        points are not equally spaced. Therefore to integrate quantities in the space, one
        should use Gauss quadrature scheme or better the Clenshaw-Curtis scheme.

        Test: PASSED 2013.8.16.
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
        if os.path.exists(config.grid.field_data):
            mat = loadmat(config.grid.field_data)
            self.wA = mat['wA']
            self.wB = mat['wB']
        else:
            self.wA = np.random.rand(Lx,Ly,Lz) - 0.5
            self.wB = np.random.rand(Lx,Ly,Lz) - 0.5
        self.qA = np.zeros((MsA, Lx, Ly, Lz))
        self.qA[0,:,:,:] = 1.
        self.qAc = np.zeros((MsA, Lx, Ly, Lz))
        self.qB = np.zeros((MsB, Lx, Ly, Lz))
        self.qBc = np.zeros((MsB, Lx, Ly, Lz))
        self.qBc[0,:,:,:] = 1.
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
        self.qA_solver = ETDRK4FxyCz(La, Lb, Lc, Lx, Ly, Lz-1,
                                             MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qAc_solver = ETDRK4FxyCz(La, Lb, Lc, Lx, Ly, Lz-1,
                                              MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qB_solver = ETDRK4FxyCz(La, Lb, Lc, Lx, Ly, Lz-1,
                                             MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.qBc_solver = ETDRK4FxyCz(La, Lb, Lc, Lx, Ly, Lz-1,
                                              MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamY = config.grid.lam[2]
        self.yita = self.lamY # For compressible model
        #self.yita = np.zeros([Lx,Ly,Lz]) # For incompressible model

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
        ii = np.arange(self.Lz)
        z = np.cos(np.pi * ii / (self.Lz - 1))
        z = 0.5 * (z + 1) * self.Lc
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
                plt.imshow(self.qB[0,:,self.Ly/2,:])
                plt.show()
                plt.imshow(self.qB[0,:,:,self.Lz/2])
                plt.show()
            self.qB_solver.solve(self.wB, self.qB[0], self.qB)
            if t % display_interval == 0:
                plt.imshow(self.qB[-1,self.Lx/2])
                plt.show()
                plt.imshow(self.qB[-1,:,self.Ly/2,:])
                plt.show()
                plt.imshow(self.qB[-1,:,:,self.Lz/2])
                plt.show()

            self.qBc_solver.solve(self.wB, self.qBc[0], self.qBc)

            self.qAc[0] = self.qBc[-1]
            if t % display_interval == 0:
                plt.imshow(self.qAc[0,self.Lx/2])
                plt.show()
                plt.imshow(self.qAc[0,:,self.Ly/2,:])
                plt.show()
                plt.imshow(self.qAc[0,:,:,self.Lz/2])
                plt.show()
            self.qAc_solver.solve(self.wA, self.qAc[0], self.qAc)
            if t % display_interval == 0:
                plt.imshow(self.qAc[-1,self.Lx/2])
                plt.show()
                plt.imshow(self.qAc[-1,:,self.Ly/2,:])
                plt.show()
                plt.imshow(self.qAc[-1,:,:,self.Lz/2])
                plt.show()

            # Calculate Q
            # Q = (1/La/Lb) * \int_0^Lx \int_0^Ly q(x,y,s=1) dx dy
            Qx = np.mean(self.qB[-1], axis=0) # integrate along x
            Qxy = np.mean(Qx, axis=0) # integrate along y
            Q = 0.5 * cheb_quadrature_clencurt(Qxy) # integrate along z
            Qcx = np.mean(self.qAc[-1], axis=0) # integrate along x
            Qcxy = np.mean(Qcx, axis=0) # integrate along x
            Qc = 0.5 * cheb_quadrature_clencurt(Qcxy)

            # Calculate density
            phiA0 = phiA
            phiB0 = phiB
            phiA = calc_density_3d(self.qA, self.qAc, self.ds) / Q
            phiB = calc_density_3d(self.qB, self.qBc, self.ds) / Q
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
            F1xy = np.mean(F1x, axis=0)
            F1 = 0.5 * cheb_quadrature_clencurt(F1xy)
            F2 = -np.log(Q)
            F = F1 + F2

            if t % display_interval == 0:
                phiAB = phiA - phiB
                plt.imshow(phiAB[self.Lx/2])
                plt.show()
                plt.imshow(phiAB[:,self.Ly/2,:])
                plt.show()
                plt.imshow(phiAB[:,:,self.Lz/2])
                plt.show()
                plt.plot(z, phiA[self.Lx/2,self.Ly/2])
                plt.plot(z, phiB[self.Lx/2,self.Ly/2])
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
                phiA_meanxy = np.mean(phiA_meanx, axis=0)
                phiA_mean = 0.5 * cheb_quadrature_clencurt(phiA_meanxy)
                phiB_meanx = np.mean(phiB, axis=0)
                phiB_meanxy = np.mean(phiB_meanx, axis=0)
                phiB_mean = 0.5 * cheb_quadrature_clencurt(phiB_meanxy)
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


class SlabABgC3d(object):
    '''
        Self-consistent field theory calculation of diblock copolymers confined
        in a slab in 3D. The slab is in x-y plane and its normal is in z
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
        Periodic boundary conditions are imposed on x and y directions.
            dqA/ds = (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)qA - wA qA
        with boundary condtions
            [dqA/dx + kaA qA] = 0, for z=0
            [dqA/dx + kbA qA] = 0, for z=l_z
            qA(s=0) = 1, for z in (0, l_z)
        and
            dqB/ds = (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)qB - wB qB
        with boundary condtions
            [dqB/dx + kaB qB] = 0, for z=0
            [dqB/dx + kbB qB] = 0, for z=l_z
            qB(s=0) = qA(s=1), for z in (0, l_z)
        and
            dqBc/ds = (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)qBc - wB qBc
        with boundary condtions
            [dqBc/dx + kaB qBc] = 0, for z=0
            [dqBc/dx + kbB qBc] = 0, for z=l_z
            qBc(s=0) = 1, for z in (0, l_z)
        and
            dqAc/ds = (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)qAc - wA qAc
        with boundary condtions
            [dqAc/dx + kaA qAc] = 0, for z=0
            [dqAc/dx + kbA qAc] = 0, for z=l_x
            qA(s=0) = qBc(s=1), for z in (0, l_z)
        ETDRK4 is employed to solve above MDEs. First these equations are
        transformed to Fourier space in the x direction
            dqk/ds = [d^2/dy - (k_x^2 + k_y^2)] qk - F(wq)
        where q denotes the propagator in real space, and qk is obtained by
        conducting FFT in x direction of q, and F is the x direction Fourier
        transformation.
        In the Chebyshev-Gauss-Lobatto grid, using Chebyshev collocation
        methods, the transformed equation can be written in a set of ODEs:
            dqk/ds = (D^2 - (k_x^2 + k_y^2)*I) qk - F(wq)
        where D^2 is an appropriate 2n order Chebyshev differentiation matrix
        To satisfy the incompressibility condition, we must have
            [kaA * <\phi_A(r)> + kaB * <\phi_B(r)>]_{z=0} = 0
        For weak interactions, <\phi_A(r)>|{z=0} = fA, <\phi_B(r)>|{z=0} = fB.
        So only kaA and kbA is needed, kaB and kbB can be obtained via
                kaA * fA + kaB * fB = 0
                kbA * fA + kbB * fB = 0
        (Ref: Fredrickson's book, 2006, page 194.)
        Special attentions should be paid on the z direction since the grid
        points are not equally spaced. Therefore to integrate quantities in the space, one
        should use Gauss quadrature scheme or better the Clenshaw-Curtis scheme.

        Test: NONE.
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
        Lx, Ly, Lz = config.grid.Lx, config.grid.Ly, config.grid.Lz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        La, Lb, Lc = config.uc.a, config.uc.b, config.uc.c
        self.La, self.Lb, self.Lc = La, Lb, Lc
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
            self.wA = np.random.rand(Lx, Ly, Lz)
            self.wB = np.random.rand(Lx, Ly, Lz)
            self.wC = np.random.rand(Lx, Ly, Lz)

        self.qA = np.zeros((MsA, Lx, Ly, Lz))
        self.qA[0, :, :, :] = 1.
        self.qAc = np.zeros((MsA, Lx, Ly, Lz))
        self.qB = np.zeros((MsB, Lx, Ly, Lz))
        self.qBc = np.zeros((MsB, Lx, Ly, Lz))
        self.qBc[0, :, :, :] = 1.
        self.qC = np.zeros((MsC, Lx, Ly, Lz))
        self.qC[0, :, :, :] = 1.
        self.qCc = np.zeros((MsC, Lx, Ly, Lz))

        ds = 1. / (Ms - 1)
        self.ds = ds
        lbcA = BC(self.lbc, self.lbc_vc)
        rbcA = BC(self.rbc, self.rbc_vc)
        self.kaA, self.kbA = lbcA.beta, rbcA.beta
        fA, fB = self.fA, self.fB
        lbcB = copy.deepcopy(lbcA)
        lbcB.beta = -lbcA.beta * fA / fB  # See Fredrickson Book p.194
        rbcB = copy.deepcopy(rbcA)
        rbcB.beta = -rbcA.beta * fA / fB
        lbcC = copy.deepcopy(lbcA)
        rbcC = copy.deepcopy(rbcA)

        self.qA_solver = ETDRK4FxyCz(La, Lb, Lc, Lx, Ly, Lz-1,
                                     MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qAc_solver = ETDRK4FxyCz(La, Lb, Lc, Lx, Ly, Lz-1,
                                      MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qB_solver = ETDRK4FxyCz(La, Lb, Lc, Lx, Ly, Lz-1,
                                     MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.qBc_solver = ETDRK4FxyCz(La, Lb, Lc, Lx, Ly, Lz-1,
                                      MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.qC_solver = ETDRK4FxyCz(La, Lb, Lc, Lx, Ly, Lz-1,
                                     MsC, h=ds, lbc=lbcC, rbc=rbcC)
        self.qCc_solver = ETDRK4FxyCz(La, Lb, Lc, Lx, Ly, Lz-1,
                                      MsC, h=ds, lbc=lbcC, rbc=rbcC)
        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamC = config.grid.lam[2]
        self.lamY = config.grid.lam[3]
        #self.yita = self.lamY  # compressible model
        self.yita = np.zeros((Lx, Ly, Lz))  # incompressible model

    def show_grid(self, data):
        Lx, Ly = self.Lx, self.Ly
        La, Lb, Lc = self.La, self.Lb, self.Lc
        yp, zp, dataxp = contourf_slab2d(data[Lx/2], Lb, Lc)
        plt.show()
        xp, zp, datayp = contourf_slab2d(data[:, Ly/2, :], La, Lc)
        plt.show()

        plt.plot(zp, dataxp[Ly/2])
        plt.plot(zp, datayp[Lx/2])
        plt.xlabel('$z$')
        plt.ylabel('grid data')
        plt.show()

    def show_density(self, phiA, phiB, phiC):
        Lx, Ly = self.Lx, self.Ly
        La, Lb, Lc = self.La, self.Lb, self.Lc

        yp, zp, phiAxp = contourf_slab2d(phiA[Lx/2], Lb, Lc)
        plt.savefig('phiAx.eps', format='eps', bbox_inches='tight')
        xp, zp, phiAyp = contourf_slab2d(phiA[:, Ly/2, :], La, Lc)
        plt.savefig('phiAy.eps', format='eps', bbox_inches='tight')
        yp, zp, phiBxp = contourf_slab2d(phiB[Lx/2], Lb, Lc)
        plt.ssavefig('phiBx.eps', format='eps', bbox_inches='tight')
        xp, zp, phiByp = contourf_slab2d(phiB[:, Ly/2, :], La, Lc)
        plt.savefig('phiBy.eps', format='eps', bbox_inches='tight')
        yp, zp, phiCxp = contourf_slab2d(phiC[Lx/2], Lb, Lc)
        plt.savefig('phiCx.eps', format='eps', bbox_inches='tight')
        xp, zp, phiCyp = contourf_slab2d(phiC[:, Ly/2, :], La, Lc)
        plt.savefig('phiCy.eps', format='eps', bbox_inches='tight')

        plt.plot(zp, phiAxp[Lx/2], label='$\phi_Ax$')
        plt.plot(zp, phiAyp[Ly/2], label='$\phi_Ay$')
        plt.plot(zp, phiBxp[Lx/2], label='$\phi_Bx$')
        plt.plot(zp, phiByp[Ly/2], label='$\phi_By$')
        plt.plot(zp, phiCxp[Lx/2], label='$\phi_Cx$')
        plt.plot(zp, phiCyp[Ly/2], label='$\phi_Cy$')
        plt.legend(loc='best')
        plt.xlabel('$z$')
        plt.ylabel('$\phi(z)$')
        plt.savefig('density_profile', format='eps', bbox_inches='tight')

    def run(self):
        config = self.config
        fA, fB, fC = self.fA, self.fB, self.fC
        chiN, chiACN, chiBCN = self.chiN, self.chiACN, self.chiBCN
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
        La, Lb, Lc = self.La, self.Lb, self.Lc
        lamA, lamB, lamC, lamY = self.lamA, self.lamB, self.lamC, self.lamY
        ds = self.ds
        sigma = self.sigma
        print 'fA=', fA, 'fB=', fB, 'fC=', fC
        print 'MsA=', self.config.grid.vMs[0],
        print 'MsB=', self.config.grid.vMs[1],
        print 'MsC=', self.config.grid.vMs[2]
        print 'chiAB*N=', chiN, 'chiBC*N=', chiACN, 'chiBC*N=', chiBCN
        print 'lamA=', lamA, 'lamB=', lamB, 'lamC=', lamC, 'lamY=', lamY
        print 'Lx=', Lx, 'Ly=', Ly, 'Lz=', Lz
        print 'La=', La, 'Lb=', Lb, 'Lc=', Lc
        print 'sigma=', sigma

        #if self.lbcC == DIRICHLET:
        #    iz0 = -2
        #else:
        iz0 = -1  # DBC is not supported currently
        delta, z = make_delta(Lz-1, iz0, Lc)
        print 'z0=', z[iz0], 'iz0=', iz0

        display_interval = config.scft.display_interval
        display_interval_test = 100000
        record_interval = config.scft.record_interval
        save_interval = config.scft.save_interval
        save_q = config.scft.is_save_q
        thresh_residual = config.scft.thresh_residual
        data_file = os.path.join(config.scft.base_dir,
                                 config.scft.data_file)
        q_file = os.path.join(config.scft.base_dir,
                              config.scft.q_file)

        phiA = np.zeros((Lx, Ly, Lz))
        phiB = np.zeros((Lx, Ly, Lz))
        phiC = np.zeros((Lx, Ly, Lz))
        ii = np.arange(self.Lz)
        z = np.cos(np.pi * ii / (self.Lz - 1))
        z = 0.5 * (z + 1) * self.Lc
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
            if t % display_interval_test == 0:
                self.show_grid(self.qB[0])
            self.qB_solver.solve(self.wB, self.qB[0], self.qB)
            if t % display_interval_test == 0:
                self.show_grid(self.qB[-1])

            self.qBc_solver.solve(self.wB, self.qBc[0], self.qBc)

            self.qAc[0] = self.qBc[-1]
            if t % display_interval_test == 0:
                self.show_grid(self.qAc[0])
            self.qAc_solver.solve(self.wA, self.qAc[0], self.qAc)
            if t % display_interval_test == 0:
                self.show_grid(self.qAc[-1])

            self.qC_solver.solve(self.wC, self.qC[0], self.qC)

            for i in np.arange(Lx):
                for j in np.arange(Ly):
                    self.qCc[0, i, j] = 2.0 * delta / self.qC[-1, i, j, iz0]
            if t % display_interval_test == 0:
                self.show_grid(self.qCc[0])
            self.qCc_solver.solve(self.wC, self.qCc[0], self.qCc)
            if t % display_interval_test == 0:
                self.show_grid(self.qCc[-1])

            # Calculate Q
            # Q = (1/(La*Lb*Lc)) * \int_0^Lx \int_0^Ly \int_0^Lz
            # q(x,y,z,s=1) dxdydz
            Qx_AB = np.mean(self.qB[-1], axis=0)  # integrate along x
            Qxy_AB = np.mean(Qx_AB, axis=0)  # integrate along y
            Q_AB = 0.5 * cheb_quadrature_clencurt(Qxy_AB)  # integrate along z
            Qcx_AB = np.mean(self.qAc[-1], axis=0)  # integrate along x
            Qcxy_AB = np.mean(Qcx_AB, axis=0)  # integrate along x
            Qc_AB = 0.5 * cheb_quadrature_clencurt(Qcxy_AB)
            Qx_C = np.mean(self.qC[-1], axis=0)
            Qxy_C = np.mean(Qx_C, axis=0)
            Q_C = 0.5 * cheb_quadrature_clencurt(Qxy_C)

            # Calculate density
            phiA0, phiB0, phiC0 = phiA, phiB, phiC
            c_AB = 1. / (1 + sigma*fC)
            phiA = c_AB * calc_density_3d(self.qA, self.qAc, self.ds)
            phiB = c_AB * calc_density_3d(self.qB, self.qBc, self.ds)
            c_C = sigma * c_AB * Lb
            phiC = c_C * calc_density_3d(self.qC, self.qCc, ds)
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
            ff = chiN*phiA*phiB + chiACN*phiA*phiC + chiBCN*phiB*phiC
            ff = ff - self.wA*phiA - self.wB*phiB - self.wC*phiC
            F1x = np.mean(ff, axis=0)
            F1xy = np.mean(F1x, axis=0)
            F1 = 0.5 * cheb_quadrature_clencurt(F1xy)
            F2 = -c_AB*np.log(Q_AB) - c_C*np.log(Q_C)
            F = F1 + F2

            if t % display_interval == 0:
                self.show_density(phiA, phiB, phiC)

            # Estimate error
            # incompressible model
            resA = chiN*phiB + chiACN*phiC + self.yita - self.wA
            resB = chiN*phiA + chiBCN*phiC + self.yita - self.wB
            resC = chiACN*phiA + chiBCN*phiB + self.yita - self.wC
            resY = phiA + phiB + phiC - 1.0
            # compressible model
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
                phiA_meanx = np.mean(phiA, axis=0)
                phiA_meanxy = np.mean(phiA_meanx, axis=0)
                phiA_mean = 0.5 * cheb_quadrature_clencurt(phiA_meanxy)
                phiB_meanx = np.mean(phiB, axis=0)
                phiB_meanxy = np.mean(phiB_meanx, axis=0)
                phiB_mean = 0.5 * cheb_quadrature_clencurt(phiB_meanxy)
                phiC_meanx = np.mean(phiC, axis=0)
                phiC_meanxy = np.mean(phiC_meanx, axis=0)
                phiC_mean = 0.5 * cheb_quadrature_clencurt(phiC_meanxy)
                wA_meanx = np.mean(self.wA, axis=0)
                wA_meanxy = np.mean(wA_meanx, axis=0)
                wA_mean = 0.5 * cheb_quadrature_clencurt(wA_meanxy)
                wB_meanx = np.mean(self.wB, axis=0)
                wB_meanxy = np.mean(wB_meanx, axis=0)
                wB_mean = 0.5 * cheb_quadrature_clencurt(wB_meanxy)
                wC_meanx = np.mean(self.wC, axis=0)
                wC_meanxy = np.mean(wC_meanx, axis=0)
                wC_mean = 0.5 * cheb_quadrature_clencurt(wC_meanxy)
                print t, '\ttime =', time, '\tF =', F
                print '\tQ_AB =', Q_AB, '\tQc_AB =', Qc_AB,
                print '\tQ_C =', Q_C
                print '\t<A> =', phiA_mean, '\t<B> =', phiB_mean,
                print '\t<C> =', phiC_mean
                print '\t<wA> =', wA_mean, '\t<wB> =', wB_mean,
                print '\t<wC> =', wC_mean
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

            # Update field
            self.wA = self.wA + self.lamA * resA
            self.wB = self.wB + self.lamB * resB
            self.wC = self.wC + self.lamC * resC
            self.yita = self.yita + self.lamY * resY  # incompressible model


class CylinderAB(object):
    '''
        Self-consistent field theory calculation of diblock copolymers confined
        in a cylinder (3D). We consider cylindrical coordinates.
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
        Periodic boundary conditions are imposed on \theta and z direction.
            dqA/ds = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dt^2 + d^2/dz^2)qA - wA qA
        in the domain
            [0, R] x [0, 2\pi] x [0, 2Lz]
        with boundary condtions
            [dqA/dr + kaA qA] = 0, for r=R
            qA(s=0) = 1, in the disk domain excluding boundary.
        and
            dqB/ds = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dt^2 + d^2/dz^2)qB - wB qB
        with boundary condtions
            [dqB/dr + kaB qB] = 0, for r=R
            qB(s=0) = qA(s=1), in the disk domain excluding boundary.
        and
            dqBc/ds = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dt^2 + d^2/dz^2)qBc - wB qBc
        with boundary condtions
            [dqBc/dr + kaB qBc] = 0, for r=R
            qBc(s=0) = 1, in the disk domain excluding boundary.
        and
            dqAc/ds = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dt^2 + d^2/dz^2)qAc - wA qAc
        with boundary condtions
            [dqAc/dr + kaA qAc] = 0, for r=R
            qA(s=0) = qBc(s=1), in the disk domain excluding boundary.
        ETDRK4 is employed to solve above MDEs.
        First these equations are transformed to Fourier space in the \theta direction
            dqk/ds = (d^2/dr^2 + (1/r)d/dr - (1/r^2)kt^2 - kz^2) qk - F(wq)
        where q denotes the propagator in real space, and qk is obtained by
        conducting FFT in \theta and z direction of q, and F is the \theta and z
        direction Fourier transformation.
        In the Chebyshev-Gauss-Lobatto grid, using Chebyshev collocation
        methods, the transformed equation can be written in a set of ODEs:
            dqk/ds = [(D1+D2) + (M*E1+M*E2) - M**2*(k_t^2*I) - k_z^2*I] qk - F(wq)
        where D1, D2 are submatrix of the 2nd order Chebyshev differentiation
        matrix; E1, E2 are submatrix of the 1st order Chebyshev differentiation
        matrix. The details are covered in:
            Notebook page 2013.08.16.

        To satisfy the incompressibility condition, we must have
            [kaA * <\phi_A> + kaB * <\phi_B>]_{r=R} = 0
        For weak interactions, <\phi_A>|{r=R} = fA, <\phi_B>|{r=R} = fB.
        So only kaA is needed, kaB can be obtained via
                kaA * fA + kaB * fB = 0
        (Ref: Fredrickson's book, 2006, page 194.)
        Special attentions should be paid on the r direction since the grid
        points are not equally spaced. Therefore to integrate quantities in the
        space, one should use Gauss quadrature scheme or better the Clenshaw-Curtis scheme.

        Test: PASSED 2013.8.17.
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
        Nz = config.grid.Ly
        Nr = config.grid.Lz # Nr is the number of grid points, EVEN only.
        Nr2 = Nr / 2
        self.Nt = Nt
        self.Nz = Nz
        self.Nr = Nr
        self.Nr2 = Nr2
        R = config.uc.a
        Lc = config.uc.b # length of cylinder
        self.R = R
        self.Lc = Lc
        Ms = config.grid.Ms
        MsA = config.grid.vMs[0]
        MsB = config.grid.vMs[1]
        if os.path.exists(config.grid.field_data):
            mat = loadmat(config.grid.field_data)
            self.wA = mat['wA']
            self.wB = mat['wB']
        else:
            self.wA = np.random.rand(Nt,Nz,Nr2) - 0.5
            self.wB = np.random.rand(Nt,Nz,Nr2) - 0.5
        self.qA = np.zeros((MsA, Nt, Nz, Nr2))
        self.qA[0,:,:,:] = 1.
        self.qAc = np.zeros((MsA, Nt, Nz, Nr2))
        self.qB = np.zeros((MsB, Nt, Nz, Nr2))
        self.qBc = np.zeros((MsB, Nt, Nz, Nr2))
        self.qBc[0,:,:,:] = 1.
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
        self.qA_solver = ETDRK4Cylind(R, Lc, Nr-1, Nt, Nz,
                                             MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qAc_solver = ETDRK4Cylind(R, Lc, Nr-1, Nt, Nz,
                                              MsA, h=ds, lbc=lbcA, rbc=rbcA)
        self.qB_solver = ETDRK4Cylind(R, Lc, Nr-1, Nt, Nz,
                                             MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.qBc_solver = ETDRK4Cylind(R, Lc, Nr-1, Nt, Nz,
                                              MsB, h=ds, lbc=lbcB, rbc=rbcB)
        self.lamA = config.grid.lam[0]
        self.lamB = config.grid.lam[1]
        self.lamY = config.grid.lam[2]
        self.yita = self.lamY # For compressible model
        #self.yita = np.zeros([Nt,Nz,Nr2]) # For incompressible model

    def run(self):
        config = self.config
        print 'fA=', self.fA, 'chiN=', self.chiN
        print 'Nt=', self.Nt, 'Nz=', self.Nz, 'Nr=', self.Nr
        print 'R=', self.R, 'Lc=', self.Lc
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

        phiA = np.zeros([self.Nt, self.Nz, self.Nr2])
        phiB = np.zeros([self.Nt, self.Nz, self.Nr2])
        tt = np.arange(self.Nt) * 2 * np.pi / self.Nt
        zz = np.arange(self.Nz) * 2 * np.pi / self.Nz
        ii = np.arange(self.Nr)
        rr = np.cos(np.pi * ii / (self.Nr - 1))
        r2 = self.R * rr[:self.Nr2]
        rxy, txy = np.meshgrid(r2, tt)
        x, y = rxy*np.cos(txy), rxy*np.sin(txy)
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
                scft_contourf(x, y, self.qB[0,:,self.Nz/4,:])
                scft_contourf(x, y, self.qB[0,:,self.Nz/2,:])
                scft_contourf(x, y, self.qB[0,:,self.Nz*3/4,:])
                plt.xlabel('qB[0]')
                plt.show()
            self.qB_solver.solve(self.wB, self.qB[0], self.qB)
            if t % display_interval == 0:
                scft_contourf(x, y, self.qB[-1,:,self.Nz/4,:])
                scft_contourf(x, y, self.qB[-1,:,self.Nz/2,:])
                scft_contourf(x, y, self.qB[-1,:,self.Nz*3/4,:])
                plt.xlabel('qB[-1]')
                plt.show()

            self.qBc_solver.solve(self.wB, self.qBc[0], self.qBc)

            self.qAc[0] = self.qBc[-1]
            if t % display_interval == 0:
                scft_contourf(x, y, self.qAc[0,:,self.Nz/4,:])
                scft_contourf(x, y, self.qAc[0,:,self.Nz/2,:])
                scft_contourf(x, y, self.qAc[0,:,self.Nz*3/4,:])
                plt.xlabel('qAc[0]')
                plt.show()
            self.qAc_solver.solve(self.wA, self.qAc[0], self.qAc)
            if t % display_interval == 0:
                scft_contourf(x, y, self.qAc[-1,:,self.Nz/4,:])
                scft_contourf(x, y, self.qAc[-1,:,self.Nz/2,:])
                scft_contourf(x, y, self.qAc[-1,:,self.Nz*3/4,:])
                plt.xlabel('qAc[-1]')
                plt.show()

            # Calculate Q
            # Q = (1/V) * \int_0^R \int_0^{2\pi} q(x,y,s=1) r dr d\theta
            # V = \pi R^2
            #print np.max(self.qB[-1]), np.min(self.qB[-1])
            Qt = np.mean(self.qB[-1], axis=0) # integrate along \theta
            Qtz = np.mean(Qt, axis=0) # integrate along z
            Qtz = np.hstack((Qtz, -Qtz[::-1]))
            Q = cheb_quadrature_clencurt(rr*Qtz)
            Qct = np.mean(self.qAc[-1], axis=0) # integrate along \theta
            Qctz = np.mean(Qct, axis=0) # integrate along z
            Qctz = np.hstack((Qctz, -Qctz[::-1]))
            Qc = cheb_quadrature_clencurt(rr*Qctz)

            # Calculate density
            phiA0 = phiA
            phiB0 = phiB
            phiA = calc_density_3d(self.qA, self.qAc, self.ds) / Q
            phiB = calc_density_3d(self.qB, self.qBc, self.ds) / Q
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
            F1tz = np.mean(F1t, axis=0)
            F1tz = np.hstack((F1tz,-F1tz[::-1]))
            F1 = cheb_quadrature_clencurt(rr*F1tz)
            F2 = -np.log(Q)
            F = F1 + F2

            if t % display_interval == 0:
                phiAB = phiA - phiB
                scft_contourf(x, y, phiAB[:,self.Nz/4,:])
                scft_contourf(x, y, phiAB[:,self.Nz/2,:])
                scft_contourf(x, y, phiAB[:,self.Nz*3/4,:])
                plt.xlabel('phiA-phiB')
                plt.show()
                plt.plot(r2, phiA[self.Nt/2,self.Nz/2])
                plt.plot(r2, phiB[self.Nt/2,self.Nz/2])
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
                phiA_mt = np.mean(phiA, axis=0)
                phiA_mtz = np.mean(phiA_mt, axis=0)
                phiA_mtz = np.hstack((phiA_mtz, -phiA_mtz[::-1]))
                phiA_mean = cheb_quadrature_clencurt(rr*phiA_mtz)
                phiB_mt = np.mean(phiB, axis=0)
                phiB_mtz = np.mean(phiB_mt, axis=0)
                phiB_mtz = np.hstack((phiB_mtz, -phiB_mtz[::-1]))
                phiB_mean = cheb_quadrature_clencurt(rr*phiB_mtz)
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
