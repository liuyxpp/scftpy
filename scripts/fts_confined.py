#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fts_confined
============

A script for performing FTS calculations for confined block copolymers.

Copyright (C) 2014 Yi-Xin Liu

"""

import argparse

from scftpy import SCFTConfig
from scftpy import ftsSlabAS1d

parser = argparse.ArgumentParser(description='fts_confined options')
parser.add_argument('-c', '--config',
                    default='param.ini',
                    help='the configuration file of polyorder.')

args = parser.parse_args()


def run_fts(param_file):
    config = SCFTConfig.from_file(param_file)
    d = config.grid.dimension
    model = config.model.model
    if d == 1:
        if model == 'AS-fts':
            b = ftsSlabAS1d(param_file)
        else:
            raise ValueError('Unsupported FTS model!')
    else:
        raise ValueError('FTS for this space dimension is not implemented!')

    print 'Choosing FTS model:', model
    b.run()


if __name__ == '__main__':
    run_fts(args.config)
