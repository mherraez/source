# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:52:29 2015

@author: javierseguradoescudero
"""

import numpy as np
import math

# global epsilon
# global rm
# global R

# L = 10.
# H = 15.

# diam = 2.0
# rm = diam

def V(rm, r):
    """
    Potential derivative:
    - rm. Maximum distance between colliding particles
    - r.  Distance between centers
    """
    if r <= rm:
        pot = (rm/r)-1.
    else:
        pot = 0.
    #
    return pot

def dV(rm, r):
    """
    Potential derivative:
    - rm. Maximum distance between colliding particles
    - r.  Distance between centers
    """
    if r <= rm:
        pot = -rm/r/r
    else:
        pot = 0.

    return pot

def distancePoints(c1, c2):

    return np.sqrt( (c1[0] - c2[0])*(c1[0] - c2[0]) + (c1[1] - c2[1])*(c1[1] - c2[1]) )

def energy_particles(p1, p2, fun):

    # Check if polygons intersect
    if p1.collision(p2):
        # Compute potential
        r = distancePoints(p1.center, p2.center)
        rm = 0.5 * (p1.L + p2.L)
        # if p1.shape.upper() == p2.shape.upper() == 'CIRCULAR':
        #     pot = fun(rm, r)
        # else:
            # pot = fun(rm, r) * p1.polygonly.intersection(p2.polygonly).area
        pot = fun(rm, r) * p1.polygonly.intersection(p2.polygonly).area
        return pot
    else:
        return 0.0

def potential(dispersion):
    """ List comprehension -> Does not accelerate. Try parallel python module """
    CHECKED = []  # Cache to collect already checked particles

    p = dispersion.Particles
    p_master = [pm for pm in p.values() if not pm.slave]

    energy = 0.0
    for pi in p_master:
        for j in pi.neighbours: # Look at the neighbouring particles
            if j not in CHECKED:  # Discard previously checked particles
                energy += energy_particles(pi, p[j], V)

        CHECKED.append(pi.ind)

    return energy

def d_cases(indices):
    # indices = sorted(indices)
    cases = []
    # i-cases
    for k in indices:
        cases += [(k,k,j) for j in indices if j > k]

    # j-cases
    for k in indices:
        cases += [(k,i,k) for i in indices if i < k]

    return cases

def d_potential(dispersion):

    CACHE = []  # TODO (better performance ~20%). Cache to collect already checked pairs

    keys = dispersion.Particles.keys()  # dict keys
    p = dispersion.Particles  # dict

    # n = len(k)
    # energy = np.zeros(n)
    # Energy derivative components are stored into each particle instance
    for k1 in keys:
        p[k1].dV = [0.0, 0.0]

    cases = d_cases(keys) # Combinations

    for k, i, j in cases:
        pi = p[i]
        if j in pi.neighbours:
            pj = p[j]
            ddV = energy_particles(pi, pj, dV)
            if ddV != 0.:
                xij = [pj.center[0]-pi.center[0], pj.center[1]-pi.center[1]]
                rij = distancePoints(pi.center, pj.center)
                ddV /= rij
                signo = (k==i) and -1. or 1.
                # print k, i, j, signo
                p[k].dV[0] += ddV*xij[0]*signo
                p[k].dV[1] += ddV*xij[1]*signo


    # Homogenize periodic gradients
    for i in sorted(keys):
        if not p[i].auxiliary and p[i].periodic != None:
            # Compare master-slave dV
            dV1 = p[i].dV
            dV2 = p[p[i].periodic].dV
            if abs(dV1[0])+abs(dV1[1]) > abs(dV2[0])+abs(dV2[1]):
                p[p[i].periodic].dV = dV1
            else:
                p[i].dV = dV2

    # Format output as ordered list
    # denergy = []
    denergy_master = []
    for i in sorted(keys):
        if not p[i].auxiliary:
            denergy_master += list(p[i].dV)

        # denergy += list(p[i].dV)

    # return np.array(denergy), np.array(denergy_master)
    return np.array(denergy_master)
