#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scft_brush
==========

A script for performing SCFT calculations for polymer brushes.

Copyright (C) 2012 Yi-Xin Liu

"""

import argparse

from scftpy import Brush, Brush_Dimless

parser = argparse.ArgumentParser(description='scft_brush options')
parser.add_argument('-c', '--config',
                    default='param.ini',
                    help='the configuration file of polyorder.')

args = parser.parse_args()

def run_scft(param_file):
    b = Brush(param_file)
    #b = Brush_Dimless(param_file)
    b.run()

if __name__ == '__main__':
    run_scft(args.config)

