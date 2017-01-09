# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:25:12 2015

@author: javierseguradoescudero
"""

from Potentials import d_potential, potential
from conjugate_gradient_functions import nonlinear_conjugate_gradient
from myClasses import Dispersion
import numpy as np
# from scipy.optimize import fmin_cg
# import math
import os
import time
import positions

os.chdir(r'C:\Users\miguel.herraez\Desktop\VIPPER project\_DONE potential dispersion\Animations\temp')

# Dominion size
L = 60.
H = 60.
tol = 0.2
generation = 'RSPIRAL'  # RANDOM, CHESS, SPIRAL, RSPIRAL
if False:
    # Variables: diam, N
    d_ref = 7.2
    N = 45
else:
    # Variables: diam, psi
    psi = .65
    d_ref = 5. #7.2
    N = int(psi*L*H*4/np.pi/d_ref**2)

phi_ref = 0.0
dispersion = Dispersion(L, H, N)

# kw = {'shape':'LOBULAR', 'parameters':[4,]}
# kw = {'shape':'SPOLYGONAL', 'parameters':[5, 0.4]}
# kw = {'shape':'POLYGONAL', 'parameters':[4,]}
# kw = {'shape':'ELLIPTICAL', 'parameters':[0.8,]}
kw = {'shape':'CIRCULAR'}
# kw = {'shape':'CSHAPE', 'parameters':[0.15, 135.]}

dispersion.setParticle(d_ref, 0.0, 0.0, phi=phi_ref, fixed=True, **kw)  # Particle at the origin is fixed

from testsPG import cases
test = 0

try:
    particles, (L, H) = cases(test)
    N = len(particles)
    dispersion.resetList()
    dispersion = Dispersion(L, H, N)
    for p_kw in particles:
        dispersion.setParticle(**p_kw)
except:
    Dispersion.tolerance = tol
    print 'Dispersion method:',

    if generation=='CHESSBOARD':
        # Chessboard generation
        print 'CHESSBOARD'
        import random
        tx = int(L/d_ref)
        dx = L/tx
        ty = int(H/d_ref)
        dy = L/ty
        # print 'Chessboard', dx, dy, tx, ty, N
        tiles = sum([[(i,j) for i in range(tx)] for j in range(ty)], [])
        # tiles = [item for sublist in tiles for item in sublist]

        current_tiles = [t for t in tiles]
        random.shuffle(current_tiles)
        for i in range(1,N):
            # Get one tile
            tile = current_tiles.pop()

            # Place a fiber in the tile
            p, q = tile
            # Purely random in square aligned tiles
            # x0 = dx * (p + random.random())
            # y0 = dy * (q + random.random())
            # Randomly centered in square aligned tiles
            # x0 = dx * (p + 0.2 + 0.6*random.random())
            # y0 = dy * (q + 0.2 + 0.6*random.random())
            # Randomly centered in square not aligned tiles
            x0 = dx * (p + 0.3*(q%2) + 0.2 + 0.6*random.random())
            y0 = dy * (q + 0.2 + 0.6*random.random())
            # phi = np.random.normal(loc=phi_ref, scale=10.)
            phi = 360.*np.random.random()
            d = d_ref
            dispersion.setParticle(d, x0, y0, phi=phi, **kw)

            if not current_tiles:
                current_tiles = [t for t in tiles]
                random.shuffle(current_tiles)

    elif generation=='SPIRAL':
        print 'SPIRAL'
        # Archimedean spiral
        if L>H:
            dx = d_ref
            dy = H/L*d_ref
        else:
            dy = d_ref
            dx = L/H*d_ref

        a = 0.5*d_ref/np.pi
        tmax = max(L*0.5, H*0.5)/a

        # Create spiral
        spiral = positions.archimedeanSpiral(a, tmax)
        # Generate equally spaced points
        newTheta = spiral.generatePoints(N=N)
        r, theta, x, y = spiral.angleToPoints(newTheta)
        # Translate (x_c, y_c), scale (L/H) and apply random noisy displacement
        x0 = 0.5*L + dx/d_ref * (x + d_ref/2*(2*np.random.random(N)-1))
        y0 = 0.5*H + dy/d_ref * (y + d_ref/2*(2*np.random.random(N)-1))
        for i in range(1,N):
            d = d_ref
            # d = np.random.normal(loc=d_ref, scale=d_ref*0.05) + tol
            phi = np.random.normal(loc=phi_ref, scale=360.)
            dispersion.setParticle(d, x0[i], y0[i], phi=phi, **kw)

    elif generation=='RSPIRAL':
        print 'RECTANGULAR SPIRAL'
        # Rectangular spiral
        if L>H:
            dx = d_ref
            dy = H/L*d_ref
        else:
            dy = d_ref
            dx = L/H*d_ref

        if dx>dy:
            # n = int(L/dx/2)
            n = int((L-dx)/dx/2)
        else:
            # n = int(H/dy/2)
            n = int((H-dy)/dy/2)

        # Build up spiral pieces
        rectSpiral = positions.RectSpiral(dx=dx, dy=dy)
        rectSpiral.addPiece(positions.SpiralPiece(dx, [1.,0.], 0))

        for i in range(1,n+1):
            l = 2*i - 1
            py1 = positions.SpiralPiece(    l*dy, [0.,-1.], i)
            px1 = positions.SpiralPiece((l+1)*dx, [-1.,0.], i)
            py2 = positions.SpiralPiece((l+1)*dy, [0., 1.], i)
            px2 = positions.SpiralPiece((l+2)*dx, [1., 0.], i)
            rectSpiral.addPiece(py1)
            rectSpiral.addPiece(px1)
            rectSpiral.addPiece(py2)
            rectSpiral.addPiece(px2)

        points = [rectSpiral.findPoint(s_) for s_ in np.linspace(0,0.99,N)]
        x,y = zip(*points)

        # Translate (x_c, y_c), scale (L/H) and apply random perturbations of the displacement
        x0 = np.array(x) + 0.5*L + d_ref*(2*np.random.random(N)-1)
        y0 = np.array(y) + 0.5*H + d_ref*(2*np.random.random(N)-1)
        for i in range(1,N):
            d = d_ref
            # d = np.random.normal(loc=d_ref, scale=d_ref*0.05) + tol
            phi = np.random.normal(loc=phi_ref, scale=360.)
            dispersion.setParticle(d, x0[i], y0[i], phi=phi, **kw)

    else:
        # Random generation
        print 'RANDOM'
        for i in range(1,N):
            x0 = 0.5*d_ref + (L-d_ref)*random.random()
            # x0 = L*np.random.rand()
            y0 = 0.5*d_ref + (H-d_ref)*random.random()
            # y0 = H*np.random.rand()
            d = d_ref
            # d = np.random.normal(loc=d_ref, scale=d_ref*0.05) + tol
            phi = np.random.normal(loc=phi_ref, scale=360.)
            dispersion.setParticle(d, x0, y0, phi=phi, **kw)




# Update neighbours and initial periodic particles
dispersion.update()
print dispersion
dispersion.plot(title=True, numbering=False, ion=False, show=False, save='Reference.pdf')
dispersion.plot(title=False, numbering=False, ion=False, show=False, save='ITER_000.png', ticks=False)

###########################
### Main algorithm
###########################
t0 = time.time()
# import cProfile
# cProfile.run('e = nonlinear_conjugate_gradient(d_potential, potential, dispersion, 1e-9, plot=False)')
# cProfile.run('e = nonlinear_conjugate_gradient(d_energy2_per, energy2_per, x0, 1e-9)')
nonlinear_conjugate_gradient(d_potential, potential, dispersion, 1.e-6, plot=False)
tTotal = time.time()-t0
print 'Time: {0:.1f} s'.format(tTotal)


###############################
### Postprocess
###############################
# dispersion.update(periodic=False)
print dispersion
dispersion.plot(title=True, numbering=False, margin=0.0, arrows=False, ion=True)