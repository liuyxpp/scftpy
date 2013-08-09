#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scft_bulk
=========

A script for performing SCFT calculations for block copolymers in bulk.

Copyright (C) 2012 Yi-Xin Liu

"""

import argparse

from scftpy import SCFTConfig
from scftpy import BulkAB1d, BulkAB2d, BulkAB3d

parser = argparse.ArgumentParser(description='scft_brush options')
parser.add_argument('-c', '--config',
                    default='param.ini',
                    help='the configuration file of polyorder.')

args = parser.parse_args()

def run_scft(param_file):
    config = SCFTConfig.from_file(param_file)
    d = config.grid.dimension
    if d == 1:
        b = BulkAB1d(param_file)
    elif d == 2:
        b = BulkAB2d(param_file)
    elif d == 3:
        b = BulkAB3d(param_file)
    else:
        raise ValueError('Unkonwn space dimension!')
    b.run()

if __name__ == '__main__':
    run_scft(args.config)

