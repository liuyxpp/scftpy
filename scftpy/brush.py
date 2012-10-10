# -*- coding: utf-8 -*-
"""
brush
=====

SCFT for polymer melt brush.

References
----------

Liu, Y. X. *Polymer Brush III*, Technical Reports, Fudan University, 2012. 

"""

from scftpy import SCFTConfig

__all__ = ['Brush',
          ]

class Brush(object):
    def __init__(self, cfgfile):
        self.config = SCFTConfig.from_file(cfgfile)

    def run(self):
        config = self.config
        for t in xrange(config.scft.max_iter):
            pass


