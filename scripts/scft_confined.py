#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scft_confined
=============

A script for performing SCFT calculations for confined block copolymers.

Copyright (C) 2012 Yi-Xin Liu

"""

import argparse

from scftpy import SCFTConfig
from scftpy import SlabAB2d, SlabAB3d, DiskAB, CylinderAB

parser = argparse.ArgumentParser(description='scft_brush options')
parser.add_argument('-c', '--config',
                    default='param.ini',
                    help='the configuration file of polyorder.')

args = parser.parse_args()

def run_scft(param_file):
    config = SCFTConfig.from_file(param_file)
    d = config.grid.dimension
    if d == 1:
        pass
    elif d == 2:
        b = SlabAB2d(param_file)
        #b = DiskAB(param_file)
    elif d == 3:
        #b = SlabAB3d(param_file)
        b = CylinderAB(param_file)
    else:
        raise ValueError('Unkonwn space dimension!')
    b.run()

if __name__ == '__main__':
    run_scft(args.config)

