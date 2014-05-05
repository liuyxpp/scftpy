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
from scftpy import SlabAB1d, SlabABgC1d
from scftpy import SlabAB2d, SlabABgC2d, DiskAB
from scftpy import SlabAB3d, SlabABgC3d, CylinderAB

parser = argparse.ArgumentParser(description='scft_brush options')
parser.add_argument('-c', '--config',
                    default='param.ini',
                    help='the configuration file of polyorder.')

args = parser.parse_args()


def run_scft(param_file):
    config = SCFTConfig.from_file(param_file)
    d = config.grid.dimension
    model = config.model.model
    if d == 1:
        if model == 'AB':
            b = SlabAB1d(param_file)
        elif model == 'ABgC':
            b = SlabABgC1d(param_file)
        else:
            raise ValueError('Unsupported SCFT model!')
    elif d == 2:
        if model == 'AB':
            b = SlabAB2d(param_file)
        elif model == 'ABgC':
            b = SlabABgC2d(param_file)
        elif model == 'DiskAB':
            b = DiskAB(param_file)
        else:
            raise ValueError('Unsupported SCFT model!')
    elif d == 3:
        if model == 'AB':
            b = SlabAB3d(param_file)
        elif model == 'ABgC':
            b = SlabABgC3d(param_file)
        elif model == 'CylinderAB':
            b = CylinderAB(param_file)
        else:
            raise ValueError('Unsupported SCFT model!')
    else:
        raise ValueError('Unkonwn space dimension!')

    print 'Choosing SCFT model:', model
    b.run()


if __name__ == '__main__':
    run_scft(args.config)
