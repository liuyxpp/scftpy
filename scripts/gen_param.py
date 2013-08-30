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
import glob
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
    batch_var_name = cfg.get('Batch', 'name') # name of the batch variable for directory
    name_min = cfg.getfloat('Batch', 'name_min')
    name_step = cfg.getfloat('Batch', 'name_step')
    batch_var = cfg.get('Batch', 'var') # name of the batch variable
    bmin = cfg.getfloat('Batch', 'min')
    bmax = cfg.getfloat('Batch', 'max')
    bstep = cfg.getfloat('Batch', 'step')
    param_file_name = 'param.ini'

    name_var = name_min - name_step
    for var in np.arange(bmin, bmax+bstep, bstep):
        name_var += name_step
        dirname = batch_var_name + str(name_var)
        pf = os.path.join(dirname, param_file_name) # dir + filename
        ensure_dir(pf)
        cfg.set(section, batch_var, str(var))
        with open(pf, 'w') as cfgfile:
            cfg.write(cfgfile)


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        

if __name__ == '__main__':
    gen_param(args.param_file)

