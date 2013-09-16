#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
gen_param
=========

A script for generating parameter files for scftpy.

Copyright (C) 2013 Yi-Xin Liu (lyx@fudan.edu.cn)

'''

import argparse
import os
import json
from ConfigParser import SafeConfigParser

import numpy as np

parser = argparse.ArgumentParser(description='gen_param options')
parser.add_argument('-p', '--param_file',
                    default='param.ini',
                    help='SCFT configuration file, *.ini')
args = parser.parse_args()


def gen_param(param='param.ini'):
    '''
    Generate a series of parameter files and put them in the directory
    correspnding to their main batch variable.
    The directory name is the name of the batch variable.
    Example: batch variable is DL, which is the ratio of D (the diameter of the
    disk or cylinder, or the length of the slab along the iteracting surface
    normal) and L (the characteristic length of the bulk phase).
    Then we create following directories:
        DL0.8, DL1.0, DL1.2, ...., DL3.0
    which varies DL from 0.8 to 3.0 with step 0.2.
    '''
    cfg = SafeConfigParser(allow_no_value=True)
    cfg.optionxform = str
    cfg.read(param)

    section = cfg.get('Batch', 'section')
    # name of the batch variable for directory
    batch_var_name = json.loads(cfg.get('Batch', 'name'))
    nvar = len(batch_var_name)
    name_min = np.array(json.loads(cfg.get('Batch', 'name_min')))
    name_step = np.array(json.loads(cfg.get('Batch', 'name_step')))
    if nvar != name_min.size or nvar != name_step.size:
        print 'Size of batch_var_name, name_min, and name_step unequal.'
        exit()
    # name of the batch variable
    batch_var = json.loads(cfg.get('Batch', 'var'))
    bmin = np.array(json.loads(cfg.get('Batch', 'min')))
    bmax = np.array(json.loads(cfg.get('Batch', 'max')))
    bstep = np.array(json.loads(cfg.get('Batch', 'step')))
    if (len(batch_var) != bmin.size or bmin.size != bstep.size or 
        bstep.size != bmax.size):
        print 'Size of batch_var, bmin, bmax, and bstep unequal.'
        exit()
    if nvar != len(batch_var):
        print 'Size of batch_var_name and batch_var unequal.'
        exit()
    param_file_name = 'param.ini'

    bsize = np.arange(bmin[0], bmax[0] + bstep[0], bstep[0]).size
    for i in xrange(1, nvar):
        if np.arange(bmin[i], bmax[i] + bstep[i], bstep[i]).size != bsize:
            print 'Size of each batch_var list unequal.'
            exit()
    #bvar = np.zeros(nvar, bsize)
    #for i in xrange(nvar):
    #    bvar[i] = np.arange(bmin[i], bmax[i] + bstep[i], bstep[i])

    name_var = name_min - name_step
    var = bmin - bstep
    for j in xrange(bsize):
        name_var += name_step
        var += bstep
        dirname = ''
        for i in xrange(nvar):
            if abs(name_var[i]) < 1e-10:
               name_var[i] = 0.0 
            if name_var[i] < 0.0:
                dirname += batch_var_name[i] + str(abs(name_var[i])) + 'n'
            else:
                dirname += batch_var_name[i] + str(name_var[i])
            if i < nvar - 1:
                dirname = dirname + '_'
            bvar = batch_var[i]
            if abs(var[i]) < 1e-10:
                var[i] = 0.0
            if (bvar == 'BC_coefficients_left' or 
                bvar == 'BC_coefficients_right'):
                bc = json.loads(cfg.get(section, bvar))
                bc[1] = var[i]
                cfg.set(section, bvar, str(bc))
            else:
                cfg.set(section, bvar, str(var[i]))
        pf = os.path.join(dirname, param_file_name)  # dir + filename
        ensure_dir(pf)
        with open(pf, 'w') as cfgfile:
            cfg.write(cfgfile)


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


if __name__ == '__main__':
    gen_param(args.param_file)
