# -*- coding: utf-8 -*-
"""
config
======

Parse SCFT configurations.

"""

import json
from ConfigParser import SafeConfigParser

import numpy as np

__all__ = ['SCFTConfig',
          ]

class ConfigError(Exception): pass


class ModelSection(object):
    def __init__(self, data):
        section = 'Model'
        self.name = section
        self.n_block = data.getint(section, 'n_block')
        self.N = json.loads(data.get(section, 'N'))
        self.a = json.loads(data.get(section, 'a'))
        self.chiN = json.loads(data.get(section, 'chiN'))
        self.graft_density = data.getfloat(section, 'graft_density')
        self.excluded_volume = data.getfloat(section, 'excluded_volume')

        self.check()

    def check(self):
        n = self.n_block
        if n == 1:
            n_chi = 0
        else:
            n_chi = n * (n-1) / 2
        if n != len(self.N) or n != len(self.a) or n_chi != len(self.chiN):
            raise ConfigError('Length of parameter list does not '
                              'match number of components!')


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
            self.b = None
        try:
            self.c = data.getfloat(section, 'c')
        except ValueError:
            self.c = None
        try:
            self.alpha = data.getfloat(section, 'alpha')
        except ValueError:
            self.alpha = None
        try:
            self.beta = data.getfloat(section, 'beta')
        except ValueError:
            self.beta = None
        try:
            self.gamma = data.getfloat(section, 'gamma')
        except ValueError:
            self.gamma = None
        self.N_list = json.loads(data.get(section, 'N_list'))
        self.c_list = json.loads(data.get(section, 'c_list'))


class GridSection(object):
    def __init__(self, data):
        section = 'Grid'
        self.name = section
        self.dimension = data.getint(section, 'dimension')
        self.Lx = data.getint(section, 'Lx')
        self.Ly = data.getint(section, 'Ly')
        self.Lz = data.getint(section, 'Lz')
        self.Ms = data.getint(section, 'Ms')
        self.lam = json.loads(data.get(section, 'lam'))
        self.field_data = data.get(section, 'field_data')

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


class SCFTSection(object):
    def __init__(self, data):
        section = 'SCFT'
        self.name = section
        self.base_dir = data.get(section, 'base_dir')
        self.data_file = data.get(section, 'data_file')
        self.param_file = data.get(section, 'param_file')
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


class SCFTConfig(object):
    def __init__(self, data):
        self.model = ModelSection(data)
        self.uc = UnitCellSection(data)
        self.grid = GridSection(data)
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
        if n != len(self.grid.lam):
            raise ConfigError('Length of parameter list does not '
                              'match number of components!')

