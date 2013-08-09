# -*- coding: utf-8 -*-
"""
config
======

Parse SCFT configurations.

"""

import json
from ConfigParser import SafeConfigParser

import numpy as np
from scipy.io import savemat, loadmat

from chebpy import DIRICHLET, NEUMANN, ROBIN, EPS

__all__ = ['SCFTConfig',
          ]

class ConfigError(Exception): pass


class ModelSection(object):
    def __init__(self, data):
        section = 'Model'
        self.name = section
        self.n_block = data.getint(section, 'n_block')
        self.N = data.getint(section, 'N')
        self.f = json.loads(data.get(section, 'f'))
        self.a = json.loads(data.get(section, 'a'))
        self.chiN = json.loads(data.get(section, 'chiN'))
        # In dimensionless unit = \sigma R_g^2 / C
        self.graft_density = data.getfloat(section, 'graft_density')
        # In dimensionless unit = w N^2 / R_g^3
        self.excluded_volume = data.getfloat(section, 'excluded_volume')
        self.lbc = data.get(section, 'BC_left')
        self.lbc_vc = json.loads(data.get(section, 'BC_coefficients_left'))
        self.rbc = data.get(section, 'BC_right')
        self.rbc_vc = json.loads(data.get(section, 'BC_coefficients_right'))

        self.check()

    @property
    def beta(self):
        # Assume C = 1
        return (self.excluded_volume * self.graft_density * 0.5)**(2./3)

    @property
    def z_hat(self):
        # \hat{z}, in unit of R_g.
        return (4.0 * self.excluded_volume * self.graft_density)**(1./3)

    @property
    def phi_hat(self):
        # \hat{\phi}, assumed C = 1
        return (self.graft_density / self.excluded_volume * 0.25)**(1./3)

    def check(self):
        n = self.n_block
        if n == 1:
            n_chi = 0
        else:
            n_chi = n * (n-1) / 2
        if n != len(self.f) or n != len(self.a) or n_chi != len(self.chiN):
            raise ConfigError('Length of parameter list does not '
                              'match number of components!')
        if np.sum(self.f) - 1.0 > EPS:
            raise ConfigError('Total fraction is not 1.0')
        if self.lbc == ROBIN and len(self.lbc_vc) != 3:
            raise ConfigError('Left RBC without coefficients!')
        if self.rbc == ROBIN and len(self.rbc_vc) != 3:
            raise ConfigError('Right RBC without coefficients!')

    def save_to_mat(self, matfile):
        ''' Save data to matfile. '''
        savemat(matfile, self.__dict__)


class UnitCellSection(object):
    def __init__(self, data):
        section = 'UnitCell'
        self.name = section
        self.crystal_system_type = data.get(section, 'CrystalSystemType')
        self.symmetry_group = data.get(section, 'SymmetryGroup')
        self.a = data.getfloat(section, 'a')
        try:
            self.b = data.getfloat(section, 'b')
        except ValueError:
            self.b = []
        try:
            self.c = data.getfloat(section, 'c')
        except ValueError:
            self.c = []
        try:
            self.alpha = data.getfloat(section, 'alpha')
        except ValueError:
            self.alpha = []
        try:
            self.beta = data.getfloat(section, 'beta')
        except ValueError:
            self.beta = []
        try:
            self.gamma = data.getfloat(section, 'gamma')
        except ValueError:
            self.gamma = []
        self.N_list = json.loads(data.get(section, 'N_list'))
        self.c_list = json.loads(data.get(section, 'c_list'))

    def save_to_mat(self, matfile):
        ''' Save data to matfile. '''
        savemat(matfile, self.__dict__)


class GridSection(object):
    def __init__(self, data, model):
        section = 'Grid'
        self.name = section
        self.dimension = data.getint(section, 'dimension')
        self.Lx = data.getint(section, 'Lx')
        self.Ly = data.getint(section, 'Ly')
        self.Lz = data.getint(section, 'Lz')
        self.lam = json.loads(data.get(section, 'lam'))
        self.field_data = data.get(section, 'field_data')

        # for i-th block, the number of discrete points is vMs[i].
        # 0-th block: 0..1..2..k..(vMs[0]-1)
        # i-th block: 0..1..2..k..(vMs[i]-1)
        # For i>0, the point 0 of i-th block is identical to the last point of i-1
        # (i-1)-th block.
        self.Ms = data.getint(section, 'Ms')
        L = self.Ms - 1
        n = model.n_block
        f = model.f
        vMs = np.zeros(n, int)
        for i in xrange(n-1):
            vMs[i] = int(L * f[i])
        vMs[n-1] = L - np.sum(vMs[:-1])
        self.vMs = vMs + 1

        self.check()

    def check(self):
        d = self.dimension
        dd = 0
        if self.Lx > 1:
            dd += 1
        if self.Ly > 1:
            dd += 1
        if self.Lz > 1:
            dd += 1
        if d != dd: 
            raise ConfigError('Available Lx, Ly, and Lz do not '
                              'match given dimension!')

    def save_to_mat(self, matfile):
        ''' Save data to matfile. '''
        savemat(matfile, self.__dict__)


class SCFTSection(object):
    def __init__(self, data):
        section = 'SCFT'
        self.name = section
        self.base_dir = data.get(section, 'base_dir')
        self.data_file = data.get(section, 'data_file')
        self.param_file = data.get(section, 'param_file')
        # For compatible reason, some older config files do not have 'q_file'
        if data.has_option(section, 'q_file'):
            self.q_file = data.get(section, 'q_file')
        self.min_iter = data.getint(section, 'min_iter')
        self.max_iter = data.getint(section, 'max_iter')
        self.is_display = data.getboolean(section, 'is_display')
        self.is_save_data = data.getboolean(section, 'is_save_data')
        self.is_save_q = data.getboolean(section, 'is_save_q')
        self.display_interval = data.getint(section, 'display_interval')
        self.record_interval = data.getint(section, 'record_interval')
        self.save_interval = data.getint(section, 'save_interval')
        self.thresh_H = data.getfloat(section, 'thresh_H')
        self.thresh_residual = data.getfloat(section, 'thresh_residual')
        self.thresh_incomp = data.getfloat(section, 'thresh_incomp')

    def save_to_mat(self, matfile):
        ''' Save data to matfile. '''
        savemat(matfile, self.__dict__)


class SCFTConfig(object):
    def __init__(self, data):
        self.model = ModelSection(data)
        self.uc = UnitCellSection(data)
        self.grid = GridSection(data, self.model)
        self.scft = SCFTSection(data)

        self.check()

    @classmethod
    def from_file(cls, cfgfile):
       cfg = SafeConfigParser(allow_no_value=True)
       cfg.optionxform = str
       cfg.read(cfgfile)
       return cls(cfg)

    def check(self):
        n = self.model.n_block
        if n == 1 and len(self.grid.lam) < 1:
            raise ConfigError('At least 1 lam needed!')
        if n > 1 and len(self.grid.lam) < n + 1:
            raise ConfigError('Not enough number of lam!')

    def save_to_mat(self, matfile):
        ''' Save data to matfile. '''
        data = {}
        data.update(self.model.__dict__)
        data.update(self.uc.__dict__)
        data.update(self.grid.__dict__)
        data.update(self.scft.__dict__)
        del data['name']
        savemat(matfile, data)

    def display(self):
        print self.model.__dict__
        print self.uc.__dict__
        print self.grid.__dict__
        print self.scft.__dict__

