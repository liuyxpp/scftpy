# -*- coding: utf-8 -*-
"""
util
====

Utilities for SCFT calculations.

"""

__all__ = ['quad_open4',
           'quad_semiopen4',
           'quad_semiopen3',
          ]

def quad_open4(f, dx):
    '''
    Integrate f[0..N] with open interval, (0, N).
    int_f = dx * (55/24*f_1 - 1/6*f_2 + 11/8 *f_3 + f_4 + f_5 + f_6
                  ... + f_{N-4} + 11/8*f_{N-3}- 1/6*f_{N-2} + 55/24*f_{N-1})
    '''
    N = f.size - 1
    q = 55./24 * f[1] - 1./6 * f[2] + 11./8 * f[3]
    q += 55./24 * f[N-1] - 1./6 * f[N-2] + 11./8 * f[N-3]
    for i in xrange(4, N-3):
        q += f[i]

    return q * dx


def quad_semiopen4(f, dx):
    '''
    Integrate f[0..N] with semi-open interval, (0, N].
    int_f = dx * (55/24*f_1 - 1/6*f_2 + 11/8 *f_3 + f_4 + f_5 + f_6
                  ... + f_{N-3} + 23/24*f_{N-3}- 7/6*f_{N-2} + 3/8*f_{N-1})
    '''
    N = f.size - 1
    q = 55./24 * f[1] - 1./6 * f[2] + 11./8 * f[3]
    q += 3./8 * f[N] - 7./6 * f[N-1] + 23./24 * f[N-2]
    for i in xrange(4, N-2):
        q += f[i]

    return q * dx


def quad_semiopen3(f, dx):
    '''
    Integrate f[0..N] in semi-open interval, (0, N].
    int_f = dx * (55/24*f_1 - 1/6*f_2 + 11/8 *f_3 + f_4 + f_5 + f_6
                  ... + f_{N-4} + 11/8*f_{N-3}- 1/6*f_{N-2} + 55/24*f_{N-1})
    '''
    N = f.size - 1
    q = 23./12 * f[1] + 7./12 * f[2]
    q += 5./12 * f[N] + 13./12 * f[N-1]
    for i in xrange(3, N-2):
        q += f[i]

    return q * dx

