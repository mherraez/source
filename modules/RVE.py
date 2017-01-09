# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 11:54:03 2014
@author: Miguel.Herraez

Module to generate RVE with non-circular fiber shape\n
Classes:\n
- Microstructure
- Fiber\n
\n
Functions:\n
- 
"""

# ----- START OF IMPORTS
import numpy as np
import os, random
from shapely.geometry import Polygon, Point, box
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon
from matplotlib.patches import Circle as mplCircle
from FibresGeom import fiber_shape, coord2vert
# import V_Box2D
from scipy import interpolate
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import Image
#from matplotlib.collections import PatchCollection
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Box2D library and pygame
import pygame
from pygame.locals import *
from Box2D import *


try:  # Enable speedups from GEOS library
    from shapely import speedups

    if speedups.available:
        speedups.enable()
except:
    pass

#----- END OF IMPORTS

#----- EXCEPTIONS

class NoPeriodicError(Exception): pass

class ShapeNotAvailable(Exception): pass


#----- START OF CONSTANTS DEFINITION

# Fiber sets specifications
GEOMETRY = 'Geometry'
PARAMETERS = 'Parameters'
DF = 'df'
PHI = 'Phi'
VF = 'Vf'
MATERIAL = 'Material'
FIBERMATERIAL = {1: 'CF-AS4',
                 2: 'CF-T700',
                 3: 'AF-ARAMID',
                 4: 'PF-POLYAMID',
                 5: 'GF-E',
                 6: 'VOID'}

# Fiber shapes
CIRCULAR = 'CIRCULAR'
LOBULAR = 'LOBULAR'
SPOLYGON = 'SPOLYGONAL'
POLYGON = 'POLYGONAL'
ELLIPTICAL = 'ELLIPTICAL'
OVAL = 'OVAL'
CSHAPE = 'CSHAPE'

# Generation algorithm
RANDOM = 'RANDOM'
NNA = 'NNA'
PERIODIC = 'PERIODIC'
HEXAGONAL = 'HEXAGONAL'
SQUARE = 'SQUARE'
POTENTIAL = 'POTENTIAL'
DYNAMIC = 'DYNAMIC'

# Compaction algorithm
POINT = 'POINT'
DIRECTION = 'DIRECTION'
STIRRING = 'STIRRING'
BACKWARD = 'BACKWARD'  # Outward direction
FORWARD = 'FORWARD'    # Inward direction
BOX2D = 'BOX2D'        # Physics based compaction (gravity, shaking)

# Periodicity
FULL = 'FULL'       # Full periodicity
HORIZ = 'HORIZ'     # Not horizontally periodic
VERT = 'VERT'       # Not vertically periodic
NONE = 'NONE'       # Not periodic

# Forbidden periodicities
periodicity = {FULL: [],
               HORIZ: [1, 3, 5, 6, 7, 8],
               VERT: [2, 4, 5, 6, 7, 8],
               NONE: [1, 2, 3, 4, 5, 6, 7, 8]}


## Box2D. Related variables definition
TARGET_FPS = 10
TIME_STEP = 1.0/TARGET_FPS

colors = {
        b2_staticBody  :    (255,255,255),  # white
        b2_dynamicBody :    (192,192,192),  # grey
        b2_kinematicBody :  (255,  0,255),  # magenta
        'background':       (105,105,105),  # background - dimgrey
        'silver':           (192,192,192),
        'black':            (  0,  0  ,0),
        'beige':            (200,200,179),
        'khaki':            (240,230,140),
        'cadetblue':        ( 95,158,160),
        'white':            (255,255,255),
        'grey':             (128,128,128),
        'palegreen':        (152,251,152),
        }

#----- END OF CONSTANTS DEFINITION


class Microstructure():
    """ Microstructure object.
    Microstructure generation, compaction...
    - Generates a microstructure
    """

    def __init__(self, rve_size=(0.0, 0.0), fibersets=[], gen_algorithm=RANDOM, read_microstructure='', tolerance=0.1,
                 comp_algorithm=[], optPeriod=FULL):
        """ Initialization is done:
        1) Generating a new microstructure specified by fiber sets
        2) Reading an existing microstructure from text file
        """
        self.rveSize = rve_size  # RVE dimensions (tuple)
        self.sq = []  # List of fibers
        self.tolerance = tolerance
        self.compact = True  # Initially set compact flag to True to generate fibres

        if bool(fibersets):
            # self.RVEly = box(0.0, 0.0, rve_size[0], rve_size[1])
            self.add_fiber_sets(fibersets, gen_algorithm, comp_algorithm, optPeriod)
            # print 'Fibers were added'
        elif bool(read_microstructure):
            # Read microstructure from file: fibres and RVE size
            # print 'Read Microstructure from file'
            directory, filename = os.path.split(read_microstructure)
            self.read_rve(directory=directory, filename=filename)
        # else:
        #     print 'Empty microstructure'
        #     raise AttributeError, 'Empty microstructure'

    def add_fiber_sets(self, fibersets, gen_algorithm=RANDOM, comp_algorithm=[], optPeriod=FULL):
        """ Adds new fiber sets to the current microstructure """

        timeStart = time.time()

        if PERIODIC in gen_algorithm.upper():  # regular microstructures: square and hexagonal
            myFiberSet = fibersets[0]
            cur_lf = fiber_shape(geometry=myFiberSet[GEOMETRY], parameters=myFiberSet[PARAMETERS], df=myFiberSet[DF][0])
            referenceFiber = Fiber(geometry=myFiberSet[GEOMETRY], parameters=myFiberSet[PARAMETERS], L=cur_lf,
                                   phi=myFiberSet[PHI][0], center=(0.0, 0.0), period=0, Nf=1, aux=True)
            centers = periodic_layout(self.rveSize, myFiberSet[VF], referenceFiber, gen_algorithm, self.tolerance, optPeriod)
            for i, row in enumerate(centers):
                for j, center in enumerate(row):
                    self.sq.append(Fiber(geometry=myFiberSet[GEOMETRY], parameters=myFiberSet[PARAMETERS], L=cur_lf,
                                         phi=myFiberSet[PHI][0], center=center, period=0, Nf=1))

        elif POTENTIAL in gen_algorithm.upper():  # Potential generation (J.Segurado)
            # Microstructure generation based in repulsive potential
            self.wrapPotentialGeneration(fibersets, plot=False, verbose=True)

        elif DYNAMIC in gen_algorithm.upper():  # Quasi-Dynamic generation based on Box2D library
            # Microstructure generation based in repulsive potential
            self.wrapDynamicGeneration(fibersets, optPeriod)

        else:
            ## Sort fibersets by increasing size (df)
            fibersetIndices = np.argsort(np.array([fibersets[j]['df'][0] for j in range(len(fibersets))]))
            iteration = 0
            maxiterations = 15  # Number of iterations compacting
            self.compact = True

            while self.compact and iteration <= maxiterations:

                print '________________________________________'
                nFibers = len(self.sq)
                # If desired volume fraction was not achieved -> Compact the microstructure

                if iteration <> 0:

                    if self.compact:
                        print '\nCompacting is required.'

                        if self.compact and comp_algorithm:
                            print 'Compacting RVE...'
                            tini_compact = time.time()

                            if comp_algorithm[0] == POINT:
                                # Compact towards a point
                                self.compact_RVE(point=comp_algorithm[1])

                            elif comp_algorithm[0] == DIRECTION:
                                # Compact in a direction
                                self.compact_RVE(vector=comp_algorithm[1])

                            elif comp_algorithm[0] == STIRRING:
                                # stirring algorithm
                                self.stirringAlgorithm(eff=comp_algorithm[1]/100.0)

                            elif comp_algorithm[0] == BOX2D:
                                # Box2D physics
                                from V_Box2D import compactBox2D
                                compactBox2D(self, gravity=comp_algorithm[1], shake=comp_algorithm[2], autoStop=True, tolerance=self.tolerance)

                            # self.compact_RVE(algorithm=comp_algorithm[0], point=comp_algorithm[1])
                            print 'Compaction time: %5.1f' % (time.time() - tini_compact)

                self.compact = False
                print '\nAdding fibres...'
                # Add fibersets in decreasing size
                for i in fibersetIndices[-1::-1]:
                    myFiberSet = fibersets[i]
                    print '\nAdding fiber set #%d' % (i)
                    cur_lf = fiber_shape(geometry=myFiberSet[GEOMETRY], parameters=myFiberSet[PARAMETERS],
                                         df=myFiberSet[DF][0])
                    cur_D_lf = fiber_shape(geometry=myFiberSet[GEOMETRY], parameters=myFiberSet[PARAMETERS],
                                         df=myFiberSet[DF][1])

                    # Add current fibers until specified Vf
                    if myFiberSet[VF] > 0.0:
                        myFiberSet[VF] = self.fibers_placer(geometry=myFiberSet[GEOMETRY],
                                                            parameters=myFiberSet[PARAMETERS],
                                                            material=myFiberSet[MATERIAL], cur_lf=(cur_lf,cur_D_lf),
                                                            ang=myFiberSet[PHI],
                                                            user_vf=myFiberSet[VF], algorithm=gen_algorithm,
                                                            optPeriod=optPeriod)
                        self.compact = self.compact or (myFiberSet[VF] > 0.0)
                        # print myFiberSet
                        # print self.compact

                # Do not iterate if compaction is disabled
                if not comp_algorithm: break

                if nFibers == len(self.sq):
                    break

                iteration += 1
                print 'End of iteration %d.' % iteration

        self.setPeriodicity()
        print 'Total time: %4.1f s.' % (time.time() - timeStart)
        print 'END OF ALGORITHM.\n'

    def origin_fiber(self, geometry, parameters, material, Lf, phi):
        """ Origin fiber is placed on the RVE corners """
        # angle = phi * (2 * np.random.rand() - 1)  # Random orientation
        angle = generateRandomNormal(phi[0], phi[1], notNegative=False)
        # l = np.random.normal(loc=Lf, scale=Lf / 30)  # Random fiber size
        l = generateRandomNormal(Lf[0], Lf[1])
        nf = [1, 0, 0, 0]
        period = [5, 6, 7, 8]
        center_x0 = [0, 0, self.rveSize[0], self.rveSize[0]]
        center_y0 = [0, self.rveSize[1], self.rveSize[1], 0]
        self.sq = [Fiber(geometry=geometry, parameters=parameters, material=material, L=l, phi=angle,
                         center=(center_x0[i], center_y0[i]), period=period[i], Nf=nf[i]) for i in range(4)]
        return l

    def fibers_placer(self, geometry, parameters, material, cur_lf, ang, user_vf, algorithm, optPeriod):
        """
        Add fibers until user_vf is reached or saturation
        """
        vf, n = self.fiber_volume()
        print 'Initial number of fibers', n
        area = self.rveSize[0] * self.rveSize[1]
        iteration = 0
        iter_limit = max(200, min(round(area / (self.rveSize[0] * self.rveSize[1])), 800)) # 100 < iter_limit < 500

        print 'maximum number of iterations: %g' % iter_limit
        print 'initial fiber fraction %4.2f %%' % (100 * vf)

        # Add origin fiber if the fibers list is empty
        if not bool(self.sq) and (optPeriod == FULL):
            self.origin_fiber(geometry=geometry, parameters=parameters, material=material, Lf=cur_lf, phi=ang)
            cur_vf, n = self.fiber_volume()
            user_vf -= cur_vf * 100
        else:
            cur_vf = 0.0
        k = len(self.sq)
        # Main loop to insert new polygons
        # init_time = time.time()
        # while (round(cur_vf, 2) < user_vf/100) & (iteration < iter_limit):
        while (user_vf / 100 > 0.0) & (iteration < iter_limit):
            s = k + 1
            L, phi, center = self.fiber_positioning(cur_lf, ang, algorithm, n, iteration, iter_limit)
            fiber = Fiber(geometry=geometry, parameters=parameters, material=material, L=L, phi=phi,
                          center=center, Nf=1)
            self.sq.append(fiber)
            # Check new fiber periodicity
            #print '----------Polygon: %d' % s
            period = self.periodicity_check(fiber)
            fiber.set_period(period)
            if period not in [0, 5, 6, 7, 8]:
                flagPeriod = 1
                #print 'Polygon %d is periodic, P = %d' % (s,Period)
                # Periodic fiber generation
                self.periodic_fiber(fiber)
                s += 1
            else:
                flagPeriod = 0
                #       Check position of fiber s (and period if exists)
                #       If the new fiber is not Valid (valid = 0)
                #       If the new fiber is OK (valid = 1)
            valid = self.valid_fiber(fiber, optPeriod)
            if not valid:
                # Exit loop if the RVE is collapsing
                iteration += 1
                if iteration % 100 == 0:
                    print('Iteration: %d out of %d' % (iteration, iter_limit))
                #print 'Fiber %d overlapping on iteration %d' % (s, iteration)
                # Delete not valid fiber and its periodic (if it exists)
                s = s - 1 - flagPeriod
                del self.sq[s:]
                continue
            k = s
            n += 1
            #        time_now = time.time()
            cur_vf += fiber.polygonly.area / area
            user_vf -= fiber.polygonly.area / area * 100
            # print 'Fiber %g added, current Vf = %4.2f %%' % (n, cur_vf * 100)
            #        print 'Loop %d, Vf = %4.2f, trial = %d' % (N,Vf,iteration)
            iteration = 0
        vf += cur_vf
        print 'Total number of fibers: %d' % n
        print 'Current Volume fiber = %4.2f %%' % (round(100 * cur_vf, 2))
        print 'Total Volume fiber = %4.2f %%' % (round(100 * vf, 2))
        # end_time = time.time()
        # print 'Total time = %4.2f sec.' % (end_time-init_time)
        # Returns remaining Vf (%) of fibres. It will require compaction if it is positive.
        return user_vf

    def fiber_positioning(self, lf, ang, algorithm, n, iteration, iter_limit):
        # print 'Iteration: %d' % iteration
        # print 'Algorithm: %s' % algorithm
        L = generateRandomNormal(lf[0], lf[1])
        phi = generateRandomNormal(ang[0], ang[1], notNegative=False)
        if (algorithm == RANDOM) or (iteration > 50) or (not bool(self.sq)):
            # Random position
            center = (self.rveSize[0] * np.random.rand(), self.rveSize[1] * np.random.rand())
        elif NNA in algorithm:
            # Position based on previous fiber
            if n % 2 == 0:
                prev_fibre = self.sq[-2]
                # fibre_index = -2
                # Second Nearest Neighbour
                prev_center = prev_fibre.center
                mindist = (prev_fibre.L + L) * 0.5
                dist = np.random.normal(loc=mindist+0.12, scale=0.012)
            else:
                # First Nearest Neighbour
                prev_fibre = self.sq[-1]  # todo object instead of index
                # fibre_index = -1
                prev_center = prev_fibre.center
                mindist = (prev_fibre.L + L) * 0.5
                dist = np.random.normal(loc=mindist+0.1, scale=0.010)
            theta = np.pi * (2 * np.random.rand() - 1)
            center = (prev_center[0] + dist * np.cos(theta), prev_center[1] + dist * np.sin(theta))
            # Check periodicity
            delta_x, delta_y = 0.0, 0.0
            if center[0] < 0:
                delta_x = self.rveSize[0]
            elif center[0] > self.rveSize[0]:
                delta_x = -self.rveSize[0]
            elif center[1] < 0:
                delta_y = self.rveSize[1]
            elif center[1] > self.rveSize[1]:
                delta_y = -self.rveSize[1]
            center = (center[0] + delta_x, center[1] + delta_y)
            # print 'Previous center = (%.2f, %.2f)' % prev_center
            # print 'New center = (%.2f, %.2f)' % center
        else:
            raise Exception(ValueError, 'No valid generation algorithm specified')

        return L, phi, center

    def fiber_volume(self):
        """
        Function: FIBER_VOLUME
        Fiber volume fraction (vf) and number of fibers excluding periodic (n)
        """
        area = self.rveSize[0] * self.rveSize[1]
        v = 0
        n = 0
        for fiber in self.sq:
            if fiber.Nf:
                n += 1
                d = fiber.L/fiber_shape(fiber.geometry, fiber.parameters, 1.0)
                v += 0.25*np.pi*d*d
                # v += fiber.polygonly.area

        vf = v / area
        return vf, int(n)

    def periodicity_check(self, fiber):
        """
        Function: PERIODICITY CHECK
        Indica si algún vértice del polígono está fuera del RVE\n
        El código numérico es el siguiente:\n
        - P = 0, Todos los vértices del polígono están contenidos en el RVE\n
        - P = 1, Alguno de los vértices está en x < 0. Izquierda\n
        - P = 2, Alguno de los vértices está en y > S0[1]. Superior\n
        - P = 3, Alguno de los vértices está en x > S0[0]. Derecha\n
        - P = 4, Alguno de los vértices está en y < 0. Inferior\n
        - P = 5, Fiber intersects bottom-left corner\n
        - P = 6, Fiber intersects top-left corner\n
        - P = 7, Fiber intersects top-right corner\n
        - P = 8, Fiber intersects bottom-right corner\n
        """
        left, right, top, bottom = False, False, False, False
        p = 0
        if np.any(fiber.vertices[0, :] < 0.0):
            p = 1
            left = True
        if np.any(fiber.vertices[1, :] > self.rveSize[1]):
            p = 2
            top = True
        if np.any(fiber.vertices[0, :] > self.rveSize[0]):
            p = 3
            right = True
        if np.any(fiber.vertices[1, :] < 0.0):
            p = 4
            bottom = True

        if left and bottom:
            p = 5
        elif left and top:
            p = 6
        elif right and top:
            p = 7
        elif right and bottom:
            p = 8

        # if np.any(fiber.vertices[0, :] < 0.0):
        #     P = 1
        # elif np.any(fiber.vertices[1, :] > self.rveSize[1]):
        #     P = 2
        # elif np.any(fiber.vertices[0, :] > self.rveSize[0]):
        #     P = 3
        # elif np.any(fiber.vertices[1, :] < 0.0):
        #     P = 4
        # else:
        #     return 0

        return p

    def periodic_fiber(self, fiber):
        """
        Function: PERIODIC_FIBER
        Generates a fiber that maintains periodicity regarding S0 and Period\n
        Input:\n
        - List of fibers (objects)\n
        - Size of the RVE, S0\n
        - Index of the fiber to generate its periodic\n
        """

        # # Reverse periodicity code
        # pp = np.array([3, 4, 1, 2])
        # period = pp[fiber.period - 1]
        # dx = np.array([1, 0, -1, 0])
        # dy = np.array([0, -1, 0, 1])
        # center_x = fiber.center[0] + dx[fiber.period - 1] * self.rveSize[0]
        # center_y = fiber.center[1] + dy[fiber.period - 1] * self.rveSize[1]

        (center, period) = self._getPeriodicCenter(fiber)

        # Periodic fiber creation
        fiberPer = Fiber(geometry=fiber.geometry, parameters=fiber.parameters, material=fiber.material, L=fiber.L,
                         phi=fiber.phi, center=center, period=period, Nf=0)
        self.sq.append(fiberPer)

    def valid_fiber(self, fiberk, optPeriod=FULL):
        """
        Function: VALID_FIBER
        Check position of fiber k (and periodic if exists)\n
          1) Interference with other fibers, v1\n
          2) Minimum distance with other fibers and RVE edges, v2\n
        If fiber is valid -> v = 1\n
        If fiber is not valid -> v = 0\n
        self.sq is a vector of Fibers\n
        'k' is the index (zero-index) of the current fiber\n
        """
        period = fiberk.period
        # 1) Interference with other fibers
        v1 = self.check_overlapping(fiberk)
        # Same check if periodic fiber exists
        if period not in [0, 5, 6, 7, 8]:
            fiberPer = self.findPeriodic(fiberk)
            v1 &= self.check_overlapping(fiberPer)

        # 2) Proximity to RVE edges
        # Check periodicity of fiberAux1 (L+tol, P1) and fiberAux2 (L-tol, P2)
        v2 = self.close_to_edge(fiberk)

        # 3) Check available periodicity
        v3 = not (fiberk.period in periodicity[optPeriod])
        if period not in [0, 5, 6, 7, 8]:
            v3 &= not (fiberPer.period in periodicity[optPeriod])  #self.check_overlapping(k + 1)

        # Check the fiber(s) validates both criteria
        return v1 & v2 & v3

    def check_overlapping(self, fiberk):
        """
        Function: CHECK_OVERLAPPING
        Check if fiber k overlaps with any other fiber given an absolute tolerance
        v1 = 1, fiber does not overlap
        """
        v1 = 1
        for fiberj in self.sq:
            d = np.sqrt( (fiberj.center[0] - fiberk.center[0])**2 + (fiberj.center[1] - fiberk.center[1])**2 )
            # Fibers sized is increased by tol_geom to avoid tiny gaps
            dmin = self.tolerance + np.sqrt(2) * (fiberj.L + fiberk.L)
            if (fiberj != fiberk) and (d < dmin):
                if self.clash(fiberj, fiberk):
                    # print 'Clash!'
                    v1 = 0
                    return v1
        return v1

    def close_to_edge(self, fiber):
        """
        Function: CLOSE_TO_EDGE
        Check if the gap between the fiber and one edge is too small
        v2 = 0 -> Fiber is too close to edge
        v2 = 1 -> Fiber is far enough from the edge
        """
        # Check periodicity of fiberAux1 (L+tol, P1) and fiberAux2 (L-tol, P2)
        v2 = 1
        fiberAux1 = Fiber(geometry=fiber.geometry, parameters=fiber.parameters, L=fiber.L + self.tolerance,
                          phi=fiber.phi, center=fiber.center, aux=True)
        fiberAux2 = Fiber(geometry=fiber.geometry, parameters=fiber.parameters, L=fiber.L - self.tolerance,
                          phi=fiber.phi, center=fiber.center, aux=True)
        p1 = self.periodicity_check(fiberAux1)
        p2 = self.periodicity_check(fiberAux2)
        # If P1 and P2 are different, the fiber is not valid
        if p1 != p2:
            v2 = 0
            #print 'Polygon too close to edge'
        return v2

    def clash(self, fiberj, fiberk):
        """
        Function: CLASH
        intersect = clash(j, k)
        Check if fibers j and k intersect given a certain tolerance
        """
        # Virtual enlarged fiber
        f1 = Fiber(geometry=fiberj.geometry, parameters=fiberj.parameters, L=fiberj.L + self.tolerance,
                   phi=fiberj.phi, center=fiberj.center, aux=True)
        return fiberk.polygonly.intersects(f1.polygonly)

    def save_rve(self, filename=r'NewMicrostructure', directory='', analysis=False):
        """
        Function: SAVE_RVE
        Create a text file to record the microstructure data
        Analyze microstructure to summarize fiber sets with their own information
        """
        if not bool(directory):
            directory = '.'
        if not filename.endswith('.txt'):
            filename += '.txt'

        # Make analysis
        analysis = self.analyzeFiberSets()
        vf, N = self.fiber_volume()
        if analysis:
            x0, y0 = self.analyzeCenterOfMass()
            Ix, Iy = self.analyzeMomentsOfInertia()

        # Create file
        f = open(os.path.join(directory, filename), 'w')

        # Header of file
        f.write('#\n#\n# %s\n#\n#\n' % time.asctime())
        f.write('# Fibre volume fraction = %4.2f %%\n' % (100 * vf))  # Fibre volume fraction
        f.write('# RVE size = %4.2f x %4.2f\n' % (self.rveSize[0], self.rveSize[1]))  # RVE dimensions
        f.write('# Number of fibres = %d\n#\n' % N)  # Number of full fibres
        if analysis:
            f.write('# Center of mass = (%.2f, %.2f)\n' % (x0, y0))
            f.write('# Moments of inertia. Ix = %.1f, Iy = %.1f\n' % (Ix, Iy))
        f.write('#\n#-----------------------------------------\n#\n')

        # Write analysis
        for i, group in enumerate(analysis):
            # Write geometry, parameters and material
            if group[0][1]:
                f.write('# %s %s: %s\n' % (group[0][0], group[0][1], group[0][2]))
            else:
                f.write('# %s: %s\n' % (group[0][0], group[0][2]))

            # Write summary: number of fibers and Vf
            f.write('# Number of fibers: %d (%d)\n' % (group[1]['n'], group[1]['n_all']))
            f.write('# Vf: %4.2f %%\n' % (100 * group[1]['vf']))

            # Write summary: size and orientation
            f.write('# Size: %4.2f (%+4.2f)\n' % (group[1]['l_mean'], group[1]['l_std']))
            f.write('# Phi(º): %6.2f (%+6.2f) %%\n' % (group[1]['phi_mean'], group[1]['phi_std']))

            f.write('#\n#-----------------------------------------\n#\n')

        # Fibres layout and characteristics
        f.write('# %6s %14s %8s %8s %8s %8s %4s %4s %12s %12s\n' %
                ('INDEX', 'MATERIAL', 'SIZE', 'PHI', 'x0', 'y0', 'P', 'Nf', 'SHAPE', 'PARAMETERS'))

        for i, fiber in enumerate(self.sq):
            # print i + 1, fiber.material, fiber.L, fiber.phi, fiber.center[0], \
            #          fiber.center[1], fiber.period, fiber.Nf, fiber.geometry
            f.write('%8d %14s %8.4f %+8.4f %8.4f %8.4f %4d %4d %12s' %
                    (i + 1, fiber.material, fiber.L, fiber.phi, fiber.center[0],
                     fiber.center[1], fiber.period, fiber.Nf, fiber.geometry))
            for j in range(len(fiber.parameters)):
                f.write(' %12.4f' % fiber.parameters[j])
            f.write('\n')

        f.close()
        print filename + ' was created successfully'

    def saveSimscreen_rve(self, filename=r'NewMicrostructure', directory=''):
        """
        Function: SAVE_RVE
        Create a text file to record the microstructure data in SIMSCREEN format
        Microstructure must be squared and only made of circular fibers
        """
        if not bool(directory):
            directory = '.'
        if not filename.endswith('.txt'):
            filename += '.txt'

        # Check:
        # - Square RVE
        if self.rveSize[0] != self.rveSize[1]:
            raise ValueError, 'RVE must be square'

        # Check:
        # - Only circular fibres
        shapes = [fiber.geometry.upper() for fiber in self.sq]
        if not all(shape == CIRCULAR for shape in shapes):
            raise ValueError, 'All fibers must be circular'

        # Create file
        f = open(os.path.join(directory, filename), 'w')

        # Microstructure dimension (must be square)
        f.write('%6.2f' % self.rveSize[0])

        for i, fiber in enumerate(self.sq):
            f.write('\n%8.4f %8.4f %8.4f' % (fiber.center[0], fiber.center[1], fiber.L * 0.5))

        f.close()
        print filename + ' was created successfully'

    def plot_rve(self, filename='', directory='', imageFormat='png', mute=True, numbering=False, COM=False,
                 MOI=False, save=False, show_plot=True, matrix_colour='', fibre_colour='', text_colour='black',
                 title=True, crack=None):
        """
        Function: PLOT_RVE
        Plot RVE cell layout.\n
        Requires matplotlib package in PYTHONPATH
        """
        try:
            from matplotlib.patches import Polygon as mplPolygon
            from matplotlib.patches import Rectangle
        except ImportError:
            raise ImportError, 'Cannot plot microstructure'

        if not bool(directory):
            directory = '.'

        maxDimension = float(max(self.rveSize))
        lmax = 10.0
        sizex = self.rveSize[0] / maxDimension * lmax
        sizey = self.rveSize[1] / maxDimension * lmax
        fig, ax = plt.subplots(num=None, figsize=(sizex, sizey), dpi=100, facecolor='w', edgecolor='k')

        L, H = self.rveSize
        if not matrix_colour: matrix_colour='dimgrey'
        ax.add_patch(Rectangle((0, 0), L, H, facecolor=matrix_colour))
        ax.set_xlim((0, L))
        ax.set_ylim((0, H))
        ax.set_aspect('equal')
        ax.set_xticks([0, L])
        ax.set_yticks([0, H])

        for i, fiber in enumerate(self.sq):

            if not bool(fibre_colour):
                fibercolour, textcolour, edgecolour = colourFiber(fiber.material)
            else:
                fibercolour = fibre_colour
                textcolour = text_colour
                edgecolour = 'k'

            # fiberpoly  = self.sq[i].poly
            if fiber.geometry.upper()==CIRCULAR:
                fiberpoly = mplCircle(fiber.center, radius=fiber.L*0.5, facecolor=fibercolour, edgecolor=edgecolour)
            else:
                polyvert = np.asarray(fiber.polygonly.exterior)
                fiberpoly = mplPolygon(polyvert, facecolor=fibercolour, edgecolor=edgecolour)

            ax.add_patch(fiberpoly)

            if numbering:
                ax.text(fiber.center[0], fiber.center[1], fiber.id + 1, color=textcolour, verticalalignment='center',
                        horizontalalignment='center', weight='semibold')

        if MOI or COM:
            x0, y0 = self.analyzeCenterOfMass()
            if COM:
                ax.plot(x0, y0, 'rx', mew=4, ms=10)
                ax.text(1.01*x0, 1.01*y0, '$('+'{0:.2f}'.format(x0)+', '+'{0:.2f}'.format(y0)+')$', color='black', verticalalignment='bottom', fontsize=16,
                                bbox=dict(facecolor='w', alpha=0.2, ec='y'), horizontalalignment='left')

            if MOI:
                ax.axhline(y=y0, color='r', ls='--')
                ax.axvline(x=x0, color='r', ls='-.')
                Ix, Iy = self.analyzeMomentsOfInertia()
                ax.text(1.01*L, y0, '$I_x = '+'{0:.0f}'.format(Ix)+'$', color='black', verticalalignment='center', fontsize=16,
                        bbox=dict(facecolor='w', alpha=0.5, ec='w'), horizontalalignment='left')
                ax.text(x0, -0.01*H, '$I_y = '+'{0:.0f}'.format(Iy)+'$', color='black', verticalalignment='top', fontsize=16,
                        bbox=dict(facecolor='w', alpha=0.5, ec='w'), horizontalalignment='center')

        if mute:
            # ax.axis('off')
            ax.set_title('')
            ax.set_xticks([])
            ax.set_yticks([])

        if crack:
            ax.plot([0,crack], [0.5*H, 0.5*H], 'k')

        if title:
            Vf, N = self.fiber_volume()
            titleString = filename + r'.\ $V_f='+'{0:4.2f}'.format(100.*Vf)+'\%$'
            # titleString = ('%s\n'+'$V_f=$ %4.2f%%') % (filename, 100*Vf)
            ax.set_title(titleString)

        if save:
            fig.savefig(os.path.join(directory, '.'.join([filename, imageFormat])))

        # plt.tight_layout()
        if show_plot:
            plt.ioff()
            plt.show()
        else:
            # Clear workspace
            plt.close()

        return fig, ax

    def read_rve(self, directory, filename):
        """
        Function: READ_RVE
        Read a text file to get the microstructure data
        """

        if not filename.endswith('.txt'):
            filename += '.txt'

        print 'Reading %s from %s' % (filename, directory)
        f = open(os.path.join(directory, filename), 'r')
        lines = f.readlines()
        for i, line in enumerate(lines):  # Skip first line

            # print str(i) + ' - ' + line
            if line.startswith('#'):
                if 'RVE' in line:
                    self.rveSize = ( float(line.split()[-3]), float(line.split()[-1]) )
                    print 'Size: ', self.rveSize
                    self.RVEly = box(0.0, 0.0, self.rveSize[0], self.rveSize[1])
                continue

            straux = line.split()
            try:
                material = straux[1]
                L = float(straux[2])
                phi = float(straux[3])
                x0 = float(straux[4])
                y0 = float(straux[5])
                period = int(straux[6])
                Nf = int(straux[7])
                geometry = straux[8]
                if len(straux) > 9:
                    params = [float(straux[j]) for j in range(9, len(straux))]
                else:
                    params = []
                fiber = Fiber(geometry=geometry, parameters=params, material=material, L=L, phi=phi, center=(x0, y0),
                              period=period, Nf=Nf)
                self.sq.append(fiber)
            except:  # Skip lines with
                print 'Line %d could not be read' % (i+1)
                continue
                # print straux
        f.close()

        # RVE dimensions
        if (self.rveSize == (0.0, 0.0)):
            Lx = [fibre.center[0] for fibre in self.sq if fibre.period >= 5]
            Ly = [fibre.center[1] for fibre in self.sq if fibre.period >= 5]
            self.rveSize = (max(Lx), max(Ly))
            print 'Size: ', self.rveSize
            self.RVEly = box(0.0, 0.0, self.rveSize[0], self.rveSize[1])
            # print 'Microstructure size: %6.2f %6.2f' % (self.rveSize[0], self.rveSize[1])

    def readSimscreen_rve(self, directory, filename):
        """
        Function: READSIMSCREEN_RVE
        Read a text file in Simscreen format to get the microstructure data
        """

        if not filename.endswith('.txt'):
            filename += '.txt'

        print 'Reading (SIMSCREEN) ', os.path.join(directory, filename)
        f = open(os.path.join(directory, filename), 'r')

        lines = f.readlines()
        for i, line in enumerate(lines):  # Skip first line
            if i == 0:
                # Read RVE dimensions
                self.rveSize = (float(line), float(line))
                continue
            straux = line.split()
            x0 = float(straux[0])
            y0 = float(straux[1])
            L = float(straux[2]) * 2
            fiber = Fiber(geometry=CIRCULAR, parameters=[], material='CF-AS4', L=L, phi=0.0, center=(x0, y0),
                          period=0, Nf=0)
            self.sq.append(fiber)
        f.close()

        # Check periodicity
        self.setPeriodicity()

        # print 'Microstructure size: %6.2f %6.2f' % (self.rveSize[0], self.rveSize[1])

    def compact_RVE(self, point=None, vector=None):
        """
        Function: COMPACT_RVE
        Compact fibers around a given point
        """

        nf = len(self.sq)

        L, H = self.rveSize

        # Determine if Punctual or Directional compaction:
        if point:
            # Compute distance between point and all fibres, sort them and get the indices
            indices = np.argsort(np.array([np.linalg.norm(point - np.array(self.sq[j].center)) for j in range(nf)]))

        elif vector:
            # Detect target plane (normal to vector)
            q = getQuadrant(vector)
            if q==1:  # 1st quadrant
                refPoint = (L, H)
            elif q==2:  # 2nd quadrant
                refPoint = (0.0, H)
            elif q==3:  # 3rd quadrant
                refPoint = (0.0, 0.0)
            elif q==4:  # 4th quadrant
                refPoint = (L, 0.0)
            else:
                raise ValueError, 'Incorrect definition of the vector'

            # Compute distance between fibres and virtual plane
            ux, uy = vector
            indices = np.argsort(np.array([distLine2Point(refPoint, (uy, -ux), self.sq[j].center) for j in range(nf)]))
            # print 'Quadrant:', q
            # print indices
        else:
            raise ValueError, 'Specify point or vector for compaction'

        # Auxiliary lists for periodicity management
        positionedFibers = list()
        # xx = [0.0, 1.0, 0.0, -1.0, 0.0]
        # yy = [0.0, 0.0, -1.0, 0.0, 1.0]

        for iteration, i in enumerate(indices):

            if i not in positionedFibers:
                # print '---------------------------------------'
                t0 = time.time()
                fiber = self.sq[i]
                if fiber.period in [1,2,3,4]:  # If fibre is periodic...
                    print 'Skipped periodic fiber %d' % (i)
                    # s = 0  # iterations counter
                    # # 'j' is the periodic fiber of 'i'
                    # j = self.sq.index(self.findPeriodic(self.sq[i]))
                    # newpoint = point
                    # while True:
                    #     s += 1
                    #     print 'Iteration:', s
                    #     original_center_i = self.sq[i].center
                    #     newCenter = self.moveFiber2Point(i, point=newpoint, isPeriodic=True, algorithm=FORWARD)
                    #     Period = int(self.sq[i].period)
                    #     newCenterPeriodic = (
                    #         newCenter[0] + xx[Period] * self.rveSize[0], newCenter[1] + yy[Period] * self.rveSize[1])
                    #     original_center_j = self.sq[j].center
                    #     self.sq[j].set_center(newCenterPeriodic)
                    #     # print 'newCenter:', newCenter
                    #     if self.check_overlapping(k=j):
                    #         # Move on to next fiber
                    #         print 'Time: %3.1f' % (time.time() - t0)
                    #         break
                    #     else:
                    #         # Get back to original position
                    #         self.sq[i].set_center(original_center_i)
                    #         self.sq[j].set_center(original_center_j)
                    #         i, j = j, i  # Exchange fibers to apply move2Fiber function
                    #         newpoint = newCenterPeriodic
                elif fiber.period == 0:
                    print 'Allocating non-periodic fiber %d.' % (i+1),
                    newCenter = self.moveFiber(fiber, vector=vector, point=point, isPeriodic=False)  #, algorithm=FORWARD)
                    fiber.set_center(newCenter)
                    print 'Time: %3.1f' % (time.time() - t0)
                    positionedFibers.append(i)
                    # print 'Newcenter:', newCenter
                else:
                    print 'Skipped fiber', i
            else:
                print 'skipped fiber (positioned)', i

            if ((iteration + 1) % 5 == 0) or (iteration==(nf-1)):
                print '----------------------------------> Progress: %2.0f%% (%d of %d)' % (
                    float(iteration + 1) / len(indices) * 100, iteration + 1, len(indices))

    def moveFiber(self, fiber, vector=None, point=None, isPeriodic=False, partition_step=2.0):  #, algorithm=FORWARD):
        """
        Function: MOVE_FIBER
        Move fiber towards a point or in a direction checking validity.
	    If fiber 'i' is not periodic initially the function preserves its non-periodicity

        BACKWARD: The fiber is directly moved to the point and progressively moves back until an
        available position.

        FORWARD: The fiber is moved towards the point while it finds an available position.
        """
        # 'fw' determines the direction of the movement
        # fw = (algorithm == FORWARD)

        # Initial fiber center
        x1 = fiber.center[0]
        y1 = fiber.center[1]

        if point:
            # Distance vector
            Mx = point[0] - x1
            My = point[1] - y1
        elif vector:
            Mx, My = vector
            # print Mx, My
        else:
            raise ValueError, 'Specify point or vector for compaction'

        Mr = np.sqrt(Mx ** 2 + My ** 2)
        u = (np.array([Mx, My])) / Mr  # Unit vector
        # TODO adaptive step
        dr = self.tolerance / partition_step  # Space-step

        # Place polygon i in the closest available position to the point or furthest of the origin of the fiber
        # r_ = Mr
        r1 = 0.0
        valid = 1

        iteration = 0
        # originalCenter = fiber.center
        # print originalCenter,
        while valid and (vector or point and (r1 <= Mr)):
            prevCenter = (x1 + r1 * u[0], y1 + r1 * u[1])
            # Produce small displacement of fiber i. Set new center
            r1 += dr
            newCenter = (x1 + r1 * u[0], y1 + r1 * u[1])
            fiber.set_center(newCenter)

            # Check validity
            valid = self.check_overlapping(fiber)

            if not isPeriodic:
                validPeriod = self.close_to_edge(fiber=fiber)
                valid &= validPeriod

            if not valid:
                # Fiber position not available
                print 'Iterations: %d' % iteration,
                return prevCenter
            iteration += 1
        print 'iterations: %d' % iteration,
        return newCenter

    def moveFiber2Point(self, i, point=(0.0, 0.0), isPeriodic=False, partition_step=2.0, algorithm=FORWARD):
        """
        Function: MOVE_FIBER_TO_POINT
        Move fiber towards a point checking validity.
	    If fiber 'i' is not periodic initially the function preserves its non-periodicity

        BACKWARD: The fiber is directly moved to the point and progressively moves back until an
        available position.

        FORWARD: The fiber is moved towards the point while it finds an available position.
        """
        # 'fw' determines the direction of the movement
        fw = (algorithm == FORWARD)
        # Initial fiber center
        x1 = self.sq[i].center[0]
        y1 = self.sq[i].center[1]
        # Distance vector
        Mx = point[0] - x1
        My = point[1] - y1
        Mr = np.sqrt(Mx ** 2 + My ** 2)
        u = (np.array([Mx, My])) / Mr  # Unit vector
        dr = self.tolerance / partition_step  # Space-step
        # Place polygon i in the closest available position to the point or furthest of the origin of the fiber
        if fw:
            r_ = Mr
            r1 = 0.0
            valid = 1
        else:
            r_ = Mr - self.sq[i].L / 3
            r1 = r_
            valid = 0

        while (fw and (valid and (r1 <= Mr))) or (not fw and (not valid)):
            prevCenter = fw and (x1 + r1 * u[0], y1 + r1 * u[1]) or (x1, y1)
            # Produce small displacement of fiber i. Set new center
            r1 = fw and r1 + dr or r1 - dr
            newCenter = (x1 + r1 * u[0], y1 + r1 * u[1])
            self.sq[i].set_center(newCenter)
            # Check validity
            valid = self.check_overlapping(k=i)
            if not isPeriodic:
                validPeriod = self.close_to_edge(fiber=self.sq[i])
                valid = valid and validPeriod
            if not valid:
                # Fiber position not available
                if fw:
                    self.sq[i].set_center(prevCenter)
                    return prevCenter
                else:
                    if r1 <= self.tolerance:
                        #              	    if (self.sq[i].center[0]-x1)*Mx < 0:
                        print 'Fiber %g keeps in place: %5.3f, %5.3f' % (i, x1, y1)
                        self.sq[i].set_center(prevCenter)
                        return prevCenter
        return newCenter

    def stirringAlgorithm(self, eff=1.0):
        """
        Function: STIRRING_RVE
        Fibers stirring algorithm proposed by Melro et al. 2008\n
        'Generation of random distribution of fibres in long-fibre reinforced composites'
        Nc is assumed 3 as proposed
        """
        fibers0 = [f for f in self.sq if f.period==0]
        n0 = len(fibers0)

        tini = time.time()

        for i, fiber in enumerate(fibers0):
            if not fiber.period == 0:  # Periodic fibers remain in their place
                continue

            t0 = time.time()
            # Only works for non-periodical fibres, P = 0
            Nc = 3
            N_cycles = 3
            print 'Placing fiber %g' % i,
            #last_fibers = []
            for n in range(1, N_cycles + 1):
                if n == 1:
                    # Move to closest fiber
                    fiberj = self.findClosest(fiber)
                    last_fibers = [fiberj, fiberj]
                elif bool(n % Nc):
                    # Move to 2nd closest fiber
                    fiberj = self.findClosest(fiber, f=2)
                    last_fibers.insert(0, fiberj)
                    last_fibers.pop()
                else:
                    # Move to closest fiber discarding two last used
                    fiberj = self.findClosest(fiber, exclude=last_fibers)
                    last_fibers.insert(0, fiberj)
                    last_fibers.pop()
                newcenter = self.moveFiber(fiber, point=fiberj.center)
                fiber.set_center(newcenter)
                # sq = moveFiber2Fiber(sq, i, j, tol, S0)
            print 'Time: %.2f\n' % (time.time()-t0),

            if ((i + 1) % 5 == 0) or (i==(n0-1)):
                print '----------------------------------> Progress: %2.0f%% (%d of %d)' % (
                    float(i + 1) / n0 * 100, i + 1, n0)

            if (i+1)/n0 >= eff:
                print 'Total time: %.2f' % (time.time()-tini)
                return

        print 'Total time: %.2f' % (time.time()-tini)

    def findPeriodic(self, fiber):
        """
        Function: FIND_PERIODIC
        Returns periodic fiber
        """
        x1 = fiber.center[0]
        y1 = fiber.center[1]
        Period = int(fiber.period)
        xx = [0.0, 1.0, 0.0, -1.0, 0.0]
        yy = [0.0, 0.0, -1.0, 0.0, 1.0]
        target_point = (x1 + xx[Period] * self.rveSize[0], y1 + yy[Period] * self.rveSize[1])
        point = Point(target_point)
        for fiberTarget in self.sq:
            if point.within(fiberTarget.polygonly):
                return fiberTarget
        raise NoPeriodicError, 'Cannot find periodic fibre'

    def findClosest(self, fiber, exclude=[], f=1):
        """
        Function: FIND_CLOSEST
        Finds 'f-th' closest fiber distinguishing periodic fibers
        :param fiber:
        :param exclude:
        :param f:
        :return:
        """

        if fiber.period == 0:
            return self._wrap_findClosest(fiber,exclude,f)
        elif fiber.period in [1,2,3,4]:
            fiberP = self.findPeriodic(fiber)
            fc_1 = self._wrap_findClosest(fiber,exclude,f)
            fc_2 = self._wrap_findClosest(fiberP,exclude,f)
            if fiber.polygonly.distance(fc_1.polygonly) < fiberP.polygonly.distance(fc_2.polygonly):
                return fc_1
            else:
                return fc_2

    def _wrap_findClosest(self, fiber, exclude=[], f=1):
        """
        Function: FIND_CLOSEST
        Finds 'f-th' closest fiber
        It can determine first closest, second closest, etc.
        It allows excluding fibers
        Default: first fiber and none excluded
        """
        # TODO solve periodic fibres
        nf = len(self.sq)
        # closeFibers = list()

        # Compute distance between two fibers
        d = np.array([fiber.polygonly.distance(ref.polygonly) for ref in self.sq])

        # Get indices that sort the distances
        indices = np.argsort(d)
        s = 0
        for g in range(1, nf):  # Discard self fiber
            k = indices[g]
            if self.sq[k] not in exclude:
                s += 1
            if s == f:
                return self.sq[k]
        raise ValueError, 'There are not fibres enough to compute %d-th neighbour' % f

    def orientationClosest(self, fiber, exclude=[], f=1):

        if fiber.period == 0:
            fTarget = self._wrap_findClosest(fiber,exclude,f)
        elif fiber.period in [1,2,3,4]:
            fiberP = self.findPeriodic(fiber)
            fc_1 = self._wrap_findClosest(fiber,exclude,f)
            fc_2 = self._wrap_findClosest(fiberP,exclude,f)
            if fiber.polygonly.distance(fc_1.polygonly) < fiberP.polygonly.distance(fc_2.polygonly):
                fTarget = fc_1
            else:
                fiber = fiberP
                fTarget = fc_2
        else:
            raise ValueError('Periodicity not supported')

        c1x, c1y = fiber.center
        c2x, c2y = fTarget.center

        phi = np.arctan2(c2y-c1y,c2x-c1x)*180./np.pi
        # phi = np.arctan((c2y-c1y)/(c2x-c1x))*180./np.pi

        return phi, fTarget

    def orientationsDistribution(self, f=1, plot=False):

        relativePosition = {fiber: self.orientationClosest(fiber, f=f) for fiber in self.sq if fiber.period in [0,1,4]}

        angles = zip(*relativePosition.values())[0]

        angles = np.sort(angles)
        N = len(angles)
        cdf_angles = np.array(range(N))/float(N)
        pdf_angles = np.diff(cdf_angles)/np.diff(angles)
        pdf_angles = 10*pdf_angles/np.ma.masked_invalid(pdf_angles).max()

        if plot:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(angles, cdf_angles, 'g-')
            ax.plot(angles[1:], pdf_angles, 'r-')
            ax.plot([-180,180], [0,1], 'k:')
            ax.set_xlabel(r'$\phi\, \left(^{\circ} \right)$', fontsize=24)
            ax.set_ylabel(r'$F\,\left(\phi \right)$', fontsize=24)
            ax.set_title(r'Cumulative distribution function of $\phi$', fontsize=20)
            ax.set_xlim(-180,180)
            plt.xticks(range(-180,181,90))
            ax.set_ylim(0,1)
            for t1 in ax.get_xticklabels():
                # t1.set_color(color)
                t1.set_size(fontsize=20)
            for t1 in ax.get_yticklabels():
                # t1.set_color(color)
                t1.set_size(fontsize=20)
            # place a text box in upper left in axes coords
            # ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=32,
            #            verticalalignment='top', horizontalalignment='right',
            #            bbox=dict(boxstyle='round', facecolor='w', alpha=0.1, edgecolor='w'))
            fig.tight_layout()
            plt.show()

        return angles, cdf_angles

    def analyzeFiberSets(self):
        """ Analysis of fibre sets within the microstructure
        - fibers. List of fibers
        """

        # Make fiber groups
        configs = list()
        fiberGroups = list()

        for fiber in self.sq:
            curGroup = fiber.geometry, fiber.parameters, fiber.material

            try:
                ind = configs.index(curGroup)
            except ValueError:
                # Create new group if it is not found
                configs.append(curGroup)
                fiberGroups.append([])
                ind = configs.index(curGroup)

            # Append fiber into the corresponding group
            fiberGroups[ind].append(fiber)

        # print configs
        area = self.rveSize[0] * self.rveSize[1]

        # Analyze each group
        analysis = list()
        for i, group in enumerate(fiberGroups):
            # Number of fibers
            n = sum([fiber.Nf for fiber in group])
            n_all = len(group)
            # print 'Number of fibers: %d (%d)' % (n, n_all)

            # Fiber volume fraction
            vf = sum([fiber.polygonly.area for fiber in group if fiber.Nf]) / area
            # print 'Vf: %4.1f %%' % (100*vf)

            # Fiber size
            l = np.array([fiber.L for fiber in group if fiber.Nf])
            l_mean = np.mean(l)
            l_std = np.std(l)
            # print 'Fiber size: %4.2f (%4.2f)' % (l_mean, l_std)

            # Fiber orientation
            phi = np.array([fiber.phi for fiber in group if fiber.Nf])
            phi_mean = np.mean(phi)
            phi_std = np.std(phi)
            # print 'Fiber phi: %4.2f (%4.2f)' % (phi_mean, phi_std)

            curAnalysis = {'n': n, 'n_all': n_all, 'vf': vf,
                           'l_mean': l_mean, 'l_std': l_std,
                           'phi_mean': phi_mean, 'phi_std': phi_std}

            analysis.append((configs[i], curAnalysis))

        return analysis

    def analyzeCenterOfMass(self):
        """
        Compute Center Of Mass
        """
        L, H = self.rveSize
        RVEly = Polygon([(0, 0), (L, 0), (L, H), (0, H)])

        areas = list()
        xi = list()
        yi = list()

        for fiber in self.sq:
            if fiber.period != 0:
                # Compute intersection fiber-RVE
                intersection = fiber.polygonly.intersection(RVEly)
            else:
                intersection = fiber.polygonly

            if intersection.area != 0.0:
                areas.append(intersection.area)
                xi.append(intersection.centroid.x)
                yi.append(intersection.centroid.y)

        areaTotal = sum(areas)
        x0 = sum([area * x for area, x in zip(areas, xi)]) / areaTotal
        y0 = sum([area * y for area, y in zip(areas, yi)]) / areaTotal

        return (x0, y0)

    def analyzeMomentsOfInertia(self, center=None):
        L, H = self.rveSize
        RVEly = Polygon([(0, 0), (L, 0), (L, H), (0, H)])
        if not center:
            center = (L * 0.5, H * 0.5)

        Ix, Iy = 0.0, 0.0
        for fiber in self.sq:
            try:
                if fiber.period != 0:
                    # Intersection fiber-RVEly
                        intersection = fiber.polygonly.intersection(RVEly)
                        vertices = np.array(intersection.boundary._get_coords().xy)
                else:
                    vertices = fiber.vertices
                ix, iy, _ = moment_of_inertia(vertices, center)
                Ix += ix
                Iy += iy
            except:
                pass

        return Ix, Iy

    def analyzeNearestNeighbour(self, neighbour=1, show_plot=False):
        """
        Compute nearest neighbour distribution
        :param neighbour: by default first nearest neighbour
        """
        distances = list()
        N = len(self.sq)
        # TODO paralellize computation. Optimize algorithm
        for i, fiber in enumerate(self.sq):

            if (i+1) % 100 == 0:
                print 'Calculating nearest neighbour distribution: %d of %d fibers' % (i+1, N)

            if (0 < fiber.period <= 4) and (fiber.Nf == 1):
                # Intersection fiber-RVEly
                fiberPeriodic = self.findPeriodic(fiber)
                f1 = self.findClosest(fiber, f=neighbour)
                f2 = self.findClosest(fiberPeriodic, f=neighbour)
                d1 = fiber.polygonly.distance(f1.polygonly)
                d2 = fiberPeriodic.polygonly.distance(f2.polygonly)
                d = min(d1, d2)
                # print '=============='
                # print 'Periodic 1: %d. Closest: %d. Distance: %f' % (self.sq.index(fiber), self.sq.index(f1), d1)
                # print 'Periodic 2: %d. Closest: %d. Distance: %f' % (self.sq.index(fiberPeriodic), self.sq.index(f2), d2)
            elif (fiber.period == 0):
                f1 = self.findClosest(fiber, f=neighbour)
                d = fiber.polygonly.distance(f1.polygonly)
            else:
                continue

            distances.append(d)

        distances = np.array(distances)
        # print distances
        print 'Mean: %f' % distances.mean()
        print 'Std dev: %f ' % distances.std()

        if show_plot:
            # Plot distribution
            plot_distribution(distances, title='Nearest Neighbour Distribution')

        return distances

    def analyzeDiameters(self, show_plot=False):
        diameters = np.array([f.L for f in self.sq if f.Nf])

        # print distances
        print 'Mean: %f' % diameters.mean()
        print 'Std dev: %f ' % diameters.std()

        if show_plot:
            # Plot distribution
            plot_distribution(diameters, title='Characteristic size distribution')

        return diameters

    def clearMicrostructure(self):
        """
        Deletes fibres out of the interest region.
        Reset Periods

        :return:

        """
        L, H = self.rveSize
        RVEly = Polygon([(0, 0), (L, 0), (L, H), (0, H)])

        # Delete outer fibers
        for i, fiber in enumerate(self.sq):
            # print 'Fiber %d - Area %f' % (i+1, fiber.polygonly.intersection(RVEly).area)
            if fiber.polygonly.intersection(RVEly).area == 0.0:
                print 'Fiber %d was deleted' % (i+1)
                # Remove the fiber from the list
                self.sq.remove(fiber)
                # fiber.destroy()

        # Delete fibers which miss their periodic ones (in case there are)
        for i, fiber in enumerate(self.sq):
            if (0 < fiber.period < 5) and not self.findPeriodic(fiber):
                print 'Fiber %d was deleted' % (i+1)
                self.sq.remove(fiber)
                # fiber.destroy()

        # Reset periodicity
        self.setPeriodicity()

    def secondOrder_K(self, dr=0.1, strategy=1, plot=False, factor=None, rmax=None, verbose=False):
        """
        Compute second-order K function of the microstructure avoiding edge effects by
        generating adjacent RVE tiles.
        Based on J.Segurado Thesis
        """
        L, H = self.rveSize
        if not factor and not rmax:
            factor = 1.0

        if factor and not rmax:
            rmax = factor * np.sqrt(L*L + H*H)
        elif not factor and rmax:
            rmax = rmax
        else:
            raise AttributeError(r'Specify either factor or rmax')

        Mx = int(np.ceil(rmax/L))
        My = int(np.ceil(rmax/H))
        A = L*H
        dx = Mx*L
        dy = My*H
        # myCenter = (dx, dy)
        # print L, H, Mx, My, myCenter, rmax

        # Make an array 2Mx+1 x 2My+1
        pmicro = patternedMicrostructure(self, ncols=2*Mx+1, nrows=2*My+1, plot=False, save=False)
        # pmicro.plot_rve(show_plot=True)

        countingFibers = [fiber for fiber in self.sq if fiber.Nf==1]
        N = len(countingFibers)

        # r = np.arange(dr, rmax+2*dr, dr)
        r = np.arange(dr, rmax, dr)
        I = np.zeros_like(r)
        for fiber_micro in countingFibers:
            cx, cy = fiber_micro.center
            # Find the fiber in pmicro translated (dx,dy)
            # fiber_pmicro = [f for f in pmicro.sq if f.center == (cx+dx, cy+dy)][0]
            fiber_pmicro = sorted([(f, dist2points(f.center, (cx+dx, cy+dy))) for f in pmicro.sq], key=lambda x: x[1])[0][0]
            # print cx, cy, fiber_pmicro.center
            t0 = time.time()
            Iprev = self._intensity(fiber_pmicro, r, strategy, pmicro)
            if verbose: print 'Fiber {0:d}. {1:.1f}s'.format(fiber_pmicro.id+1, time.time()-t0)
            I += Iprev

        # fibers_pmicro = []
        # for fiber_micro in countingFibers:
        #     cx, cy = fiber_micro.center
        #     # Find the fiber in pmicro translated (dx,dy)
        #     # fiber_pmicro = [f for f in pmicro.sq if f.center == (cx+dx, cy+dy)][0]
        #     fiber_pmicro = sorted([(f, dist2points(f.center, (cx+dx, cy+dy))) for f in pmicro.sq], key=lambda x: x[1])[0][0]
        #     fibers_pmicro.append(fiber_pmicro)
        # I = sum(np.array([self._intensity(f, r, strategy, pmicro) for f in fibers_pmicro]))

        if plot:
            fig, ax = plt.subplots()
            ax.plot(r, I, 'b.-')
            ax.set_xlabel(r'$r(\mu m)$', fontsize=18)
            ax.set_ylabel(r'$K(r)$', fontsize=18)
            ax.set_title(r'Second-order intensity function', fontsize=18)
            plt.show()

        return np.array(r), np.array(I)/N/N*A

    def _intensity(self, fiber, r, strategy, pmicro, verbose=False):
        myCenter = fiber.center
        rmax = max(r)

        if strategy==1: # Fibre centers
            # t0 = time.time()
            distances = np.sort([dist2points(myCenter, fiber.center) for fiber in pmicro.sq])
            # if verbose: print 'Time distances: {}'.format(time.time()-t0)
            # Loop
            i = 1 # Don't count first fibre (origin)
            I = [0]
            for r_ in r:
                I.append(I[-1])
                j = i
                for d in distances[j:]:
                    if d<=r_:
                        I[-1] += 1
                        i += 1
                    else:
                        break
                # print r, K[-1]
            I = I[1:]
            ##

        elif strategy == 2: # Smooth intensity (I~)
            # t1 = time.time()
            # Remove central fiber
            for f in pmicro.sq:
                if f.center == myCenter:
                    # print 'Delete fiber'
                    pmicro.sq.remove(f)
                    # pmicro.plot_rve()
                    break

            # Lists to keep track of the fibers
            containedFibers = []
            partialFibers = []
            cache = {}

            I = []
            for r_ in r:
                # K.append(K[-1])

                # Update fully contained fibers
                newContainedFibers = [f for f in pmicro.sq if (dist2points(myCenter, f.center) < r_+0.5*f.L)
                                      and f not in containedFibers]
                for f in newContainedFibers:
                    cache[f] = 1.
                    if f in partialFibers: del cache[f]

                # Insert new partial fibers in the container
                newPartialFibers = [f for f in pmicro.sq if (r_-0.5*f.L < dist2points(myCenter, f.center) < r_+0.5*f.L)
                                      and f not in partialFibers]
                partialFibers += newPartialFibers
                for f in partialFibers:
                    cache[f] = circularIntersection(myCenter, r_, f).area / (f.polygonly.area)

                I.append( sum(cache.values()) )
            # print 'Time: {0:.3f} s.'.format(time.time()-t1)
            # if verbose: print 'Time K: {}'.format(time.time()-t1)
            # I = I[1:]
            ##

        elif strategy == 3: # Smooth intensity (I~) enhanced

            # Find fibers at range from the reference fiber
            fiberDistances = {f:(dist2points(myCenter, f.center)-f.L/2, dist2points(myCenter, f.center)+f.L/2)
                         for f in pmicro.sq
                         if 1.e-3 < dist2points(myCenter, f.center)-f.L/2 < rmax}
            interestFibers = set({f for f in fiberDistances.keys()})

            # Lists to keep track of the fibers
            containedFibers = set()  # Set
            partialFibers = set()  # Set
            # cache = {}

            I = np.zeros_like(r)
            if verbose: print '{0:4s} {1:3s} {2:3s} {3:3s} {4:3s} {5:3s}'.format('r', 'int', 'cont', 'part', 'newcont', 'newpart')
            for i, r_ in enumerate(r):
                # Update fully contained fibers
                newContainedFibers = {f for f in partialFibers if r_>fiberDistances[f][1]}
                # newContainedFibers = {f for f in interestFibers.union(partialFibers) if r_>fiberDistances[f][1]}

                # Update fiber sets
                containedFibers = containedFibers.union(newContainedFibers)
                interestFibers = interestFibers.difference(newContainedFibers)
                partialFibers = partialFibers.difference(newContainedFibers)
                I[i] = len(containedFibers)

                # Update new partial fibers in the container
                newPartialFibers = {f for f in interestFibers if r_>fiberDistances[f][0]}
                partialFibers = partialFibers.union(newPartialFibers)
                interestFibers = interestFibers.difference(newPartialFibers)

                if verbose: print '{0:4.2f} {1:3d} {2:3d} {3:3d} {4:3d} {5:3d}'.format(r_, len(interestFibers),
                                                                         len(containedFibers), len(partialFibers),
                                                                         len(newContainedFibers), len(newPartialFibers) )

                I[i] += sum([circularIntersection(myCenter, r_, f).area / (f.polygonly.area) for f in partialFibers])
            ##

        elif strategy == 4: # Smooth intensity (I~) annular parallelization
            # This strategy is slower and is not totally accurate
            # Find fibers at range from the reference fiber
            fiberDistances = {f:(dist2points(myCenter, f.center)-f.L/2, dist2points(myCenter, f.center)+f.L/2)
                         for f in pmicro.sq
                         if 1.e-3 < dist2points(myCenter, f.center)-f.L/2 < rmax}
            interestFibers = set({f for f in fiberDistances.keys()})

            # I = np.zeros_like(r)

            # if verbose: print '{0:4s} {1:3s} {2:3s} {3:3s} {4:3s} {5:3s}'.format('r', 'int', 'cont', 'part', 'newcont', 'newpart')
            # dr = rmax / len(r)
            dr = r[1]-r[0]
            g = [annular_intersection(myCenter, r_, dr, interestFibers, fiberDistances) for r_ in r]
            I = np.cumsum(g)
            ##
        else:
            raise NotImplementedError(r'Choose an available strategy: 1 (discrete), 2 (continuous), 3 (continuous enhanced), 4 (continuous annular)')

        return np.array(I)

    def _Eintensity(self, fiber, r, strategy, verbose=False):
        """
        Computes intensity function of fiber correcting edge effect
        :param fiber: reference fiber (circles center)
        :param r: 1-D array (or list) of radii values
        :param strategy: 1: discrete. 2: continuous
        :return: intensity 1-D array
        """

        myCenter = fiber.center
        x0, y0 = myCenter
        L, H = self.rveSize

        if strategy==1: # Fibre centers
            # t0 = time.time()
            distances = np.sort([dist2points(myCenter, fiber.center) for fiber in self.sq])
            # if verbose: print 'Time distances: {}'.format(time.time()-t0)
            # print distances
            # List comprehension
            # K = [len(np.where( distances <= r ))-1 for r in rs]
            # return rs, K

            # Loop
            i = 1 # Don't count first fibre (origin)
            I = [0]
            for r_ in r:
                I.append(I[-1])
                j = i
                for d in distances[j:]:
                    if d<=r_:
                        # Check edge intersection with the circle
                        if (x0-r_ < 0.) or (x0+r_ > L) or (y0-r_<0.) or (y0+r_>H):
                            # Correct edge effect
                            circ = Point(myCenter).buffer(r_)
                            inters = self.RVEly.intersection(circ)
                            w = inters.area/(np.pi*r_*r_)
                            if verbose: print 'Correction. Fiber {0:d}, w={1:.3f}'.format(fiber.id, w)
                        else:
                            w = 1.

                        I[-1] += 1/w
                        i += 1
                    else:
                        break
                # print r, K[-1]
            I = I[1:]
            ##

        elif strategy ==2:
            pass

        return np.array(I)

    def radialDistFunction(self, rmax, dr, plot=False, verbose=False, save='', strategy=1):
        """
        Computes the Radial Distribution Function (G) of a microstructure.
        :param rmax: maximum radial distance
        :param dr: radial step
        :param strategy: determines the methodology to compute the RDF. Strategy = 1, based on the
        numerical derivative of the intensity function (K). Strategy = 2, average of g(r).
        :param plot:
        :param verbose:
        :return:
        """

        L, H = self.rveSize
        countingFibers = [fiber for fiber in self.sq if fiber.Nf==1]
        N = len(countingFibers)
        r, K = self.secondOrder_K(dr=dr, rmax=rmax, verbose=verbose, strategy=strategy)

        # Compare with Poisson distribution
        Lp = np.sqrt(K/np.pi)-r

        # Na = N/(L*H)
        dK = np.diff(K)/np.diff(r)
        dK = np.insert(dK, 0, 0)
        g1 = dK / (2*np.pi*r)
        # Interpolate as cubic spline (smoothing)
        # tck = interpolate.splrep(r1, g1, s=0)
        # r1_new = np.arange(dr, rmax+2*dr, dr/10)
        # g1_new = interpolate.splev(r1_new, tck, der=0)

        # r, g2 = self.pairDistribFunction(dr=dr, rmax=rmax, verbose=verbose)

        # if save:
        #     f = open(save)
        #
        #     f.close()

        if plot:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(7, 8), dpi=100, facecolor='w', edgecolor='k')

            ax[0].plot(r, K, 'b', lw=2., label='Microstructure')
            ax[0].plot(r, np.pi*r**2, 'r--', lw=2., label='Poisson dist.')
            ax[0].set_xlim(0, max(r))
            ax[0].set_ylabel(r'$K(r)$', fontsize=20)
            ax[0].set_title(r'Second-order intensity function')
            leg0 = ax[0].legend(loc=2)
            leg0.draggable()

            ax[1].plot(r, g1, 'b', label='Microstructure')
            # ax[1].plot(r, g2, 'c', label='Microstructure')
            ax[1].plot(r, Lp, 'g:')
            ax[1].set_xlim(0, r[-1])
            ax[1].set_ylim([0, 1.1*max(g1)])
            ax[1].set_xlabel(r'$r\ (\mu m)$', fontsize=20)
            ax[1].set_ylabel(r'$G(r)$', fontsize=20)
            ax[1].set_title(r'Radial distribution function')
            ax[1].axhline(y=1.0, xmin=0, xmax=r[-1], color='r', ls='--', lw=2., label='Reference')
            leg1 = ax[1].legend()
            leg1.draggable()

            plt.show()

        return r, g1, K

    def pairDistribFunction(self, dr=0.1, rmax=100., verbose=False):
        """
        Compute Radial Distribution Function (aka Pair Distributino Function) of the microstructure avoiding edge effects by
         generating adjacent RVE tiles.
        Based on: A.R.Melro, P.P.Camanho, S.T.Pinho, Generation of random distribution fibres in long-fibre
         reinforced composites. Composites Science and Technology 68, pp. 2092-2102, 2008.
        """
        L, H = self.rveSize
        Mx = int(np.ceil(rmax/L))
        My = int(np.ceil(rmax/H))
        dx = Mx*L
        dy = My*H

        # Make an array 2Mx+1 x 2My+1
        pmicro = patternedMicrostructure(self, ncols=2*Mx+1, nrows=2*My+1, plot=False, save=False)

        countingFibers = [fiber for fiber in self.sq if fiber.Nf==1]
        N = len(countingFibers)
        Na = N/(L*H)

        r = np.arange(dr, rmax+2*dr, dr)
        n = np.zeros_like(r)
        for fiber_micro in countingFibers:
            cx, cy = fiber_micro.center
            # Find the fiber in pmicro translated (dx,dy)
            # fiber_pmicro = [f for f in pmicro.sq if f.center == (cx+dx, cy+dy)][0]
            fiber_pmicro = sorted([(f, dist2points(f.center, (cx+dx, cy+dy))) for f in pmicro.sq], key=lambda x: x[1])[0][0]
            # print cx, cy, fiber_pmicro.center
            t0 = time.time()
            n_prev = np.diff(self._intensity(fiber_pmicro, r, 1, pmicro))
            n_prev = np.insert(n_prev,0,0)
            if verbose: print 'Fiber {0:d}. {1:.1f}s'.format(fiber_pmicro.id+1, time.time()-t0)
            n += n_prev

        g = n / (Na*N*2*np.pi*r*dr)

        return r, g

    def setPeriodicity(self):
        """
        Assign periodicity on each fibre and corrects Nf

            6_______2_______7
            |               |
            |               |
           1|       0       |3
            |               |
            |               |
            |_______________|
            5       4       8
        """
        L, H = self.rveSize
        RVEly = Polygon([(0, 0), (L, 0), (L, H), (0, H)])
        for fiber in self.sq:
            left, top, right, bottom = False, False, False, False
            # Check periodicity
            p = 0
            if np.any(fiber.vertices[0, :] < 0.0):
                p = 1
                left = True
            if np.any(fiber.vertices[1, :] > H):
                p = 2
                top = True
            if np.any(fiber.vertices[0, :] > L):
                p = 3
                right = True
            if np.any(fiber.vertices[1, :] < 0.0):
                p = 4
                bottom = True

            if left and bottom:
                p = 5
            elif left and top:
                p = 6
            elif right and top:
                p = 7
            elif right and bottom:
                p = 8

            # Check Nf
            if p in [0, 5]:
                Nf = 1
            elif p in [6, 7, 8]:
                Nf = 0
            else:
                if Point(fiber.center).within(RVEly):
                    Nf = 1
                else:
                    Nf = 0

            # Assign period and Nf
            fiber.set_Nf(Nf=Nf)
            fiber.set_period(period=p)

        # Recheck periodic fibres, only one of them has Nf=1
        for fiber in self.sq:
            if fiber.period in [1,2,3,4]:
                fiberP = self.findPeriodic(fiber)
                if (fiber.Nf+fiberP.Nf==1):
                    continue
                elif (fiber.Nf+fiberP.Nf==0):
                    fiber.Nf = 1
                elif (fiber.Nf+fiberP.Nf==2):
                    print 'Check fibers Nf'
                    fiber.Nf = 0
                else:
                    raise NotImplementedError, 'Bad identification'

    def validate(self):
        """
        Check possible overlapping of fibers.
            If a fiber overlaps, we reduce its size as much as necessary.
        Periodic fibers are accurately made periodic
        """
        self.tolerance = 0.05

        for i, fiber in enumerate(self.sq):
            if fiber.period == 0:
                # print '-Regular fiber %d' % (i+1)
                self._resizeFiber(fiber)
            elif fiber.period in [1,2,3,4]:
                # print '-Periodic fiber %d' % (i+1)
                try:
                    fiberP = self.findPeriodic(fiber)
                    self._alignFibers(fiber, fiberP)
                    # fiberP.set_center(fiber.center)
                    self._resizeFiber(fiber)
                    fiberP.set_size(fiber.L)

                    # What if fiber (or fiberP) is not periodic any more.
                    # Redefine periodicity and delete twin fiber
                    if self.periodicity_check(fiber) == 0:
                        fiber.set_period(0)
                        fiber.set_Nf(1)
                        print 'Removed: '+str(fiberP)
                        self.sq.remove(fiberP)
                        if not self.close_to_edge(fiber): # What if fiber or fiberP is too close to edge
                            fiber.set_size(fiber.L-self.tolerance*0.5)

                    elif self.periodicity_check(fiberP) == 0:
                        fiberP.set_period(0)
                        fiberP.set_Nf(1)
                        self.sq.remove(fiber)
                        print 'Removed: '+str(fiber)
                        if not self.close_to_edge(fiberP): # What if fiber or fiberP is too close to edge
                            fiberP.set_size(fiberP.L-self.tolerance*0.5)
                except NoPeriodicError, e:
                    # Delete fiber and its periodic fiber
                    self.sq.remove(fiber)
                    self.sq.remove(fiberP)
                    print 'Could not validate -> Fiber deleted ({0:d})'.format(fiber.period)

    def localVolumeFraction(self, spotSize, voxelSize, spots=True, fibres=True, plot=True):

        L, H = self.rveSize
        Mx = int(np.ceil(spotSize/L))
        My = int(np.ceil(spotSize/H))

        # Make an array Mx x My
        pmicro = patternedMicrostructure(self, ncols=2*Mx+1, nrows=2*My+1, plot=False, save=False)

        # Create grid (cells)
        dx = dy = voxelSize
        nx = np.ceil(L/dx)+1
        xv = np.linspace(Mx*L, (Mx+1)*L, nx)
        ny = np.ceil(H/dy)+1
        yv = np.linspace(My*H, (My+1)*H, ny)
        # nx = np.ceil(L/dx)
        # xv = np.linspace(Mx*L+dx/2, (Mx+1)*L-dx/2, nx)
        # ny = np.ceil(H/dy)
        # yv = np.linspace(My*H+dy/2, (My+1)*H-dy/2, ny)
        xx, yy = np.meshgrid(xv, yv)

        LVF = np.array([ [ lvf(pmicro, x, y, spotSize) for x in xv ] for y in yv ])

        # TODO write text file with contour

        ### PLOT
        if plot:
            fig, ax0 = plt.subplots(facecolor='w', edgecolor='k')
            im = ax0.contourf(xx, yy, LVF)
            fig.colorbar(im, ax=ax0)

            # Plot computation points
            if spots: ax0.plot(xx, yy, '.', ms=2, color='k')

            # Plot fibres
            if fibres:
                for i, fiber in enumerate(pmicro.sq):
                    polyvert = np.asarray(fiber.polygonly.exterior)
                    fiberpoly = mplPolygon(polyvert, facecolor='w', edgecolor='k', fill=False, lw=2)
                    ax0.add_patch(fiberpoly)

            ax0.set_xlim(Mx*L, (Mx+1)*L)
            ax0.set_ylim(My*H, (My+1)*H)
            ax0.set_aspect('equal')
            plt.show()
            ax0.set_xticks([])
            ax0.set_yticks([])
            # plt.ion()

        return xx, yy, LVF

    def _resizeFiber(self, fiber):
        valid = self.check_overlapping(fiber)
        # print 'Initial size: %.2f' % (fiber.L)
        while not valid:
            valid = self.check_overlapping(fiber)
            if fiber.L-self.tolerance*0.5 > 0:
                fiber.set_size(fiber.L-self.tolerance*0.5)
            else:
                print 'Cannot avoid overlapping. Fiber is deleted'
                self.sq.remove(fiber)
                break
        # print 'Final size: %.2f' % (fiber.L)

    def _alignFibers(self, fiber1, fiber2=None):

        center1 = fiber1.center
        (center21, period21) = self._getPeriodicCenter(fiber1)

        # If periodic fiber is not provided look for it
        if not fiber2:
            fiber2 = self.findPeriodic(fiber1)

        center2 = fiber2.center
        (center12, period12) = self._getPeriodicCenter(fiber2)

        fiber2.set_center(center21)     # Set center of fiber2 according to fiber1

        if not self.check_overlapping(fiber2):  # If fiber2 overlaps
            fiber2.set_center(center2)      # Return fiber2 to its original place
            fiber1.set_center(center12)     # Set center of fiber1 according to fiber2

    def _getPeriodicCenter(self, fiber):
        """ Find the center of the periodic fiber """

        # Reverse periodicity code
        pp = np.array([3, 4, 1, 2])
        period = pp[fiber.period - 1]

        # Find periodic center
        dx = np.array([1, 0, -1, 0])
        dy = np.array([0, -1, 0, 1])
        center_x = fiber.center[0] + dx[fiber.period - 1] * self.rveSize[0]
        center_y = fiber.center[1] + dy[fiber.period - 1] * self.rveSize[1]

        return ((center_x, center_y), period)

    def wrapPotentialGeneration(self, fibersets, plot=True, verbose=True):
        from Potentials import d_potential, potential
        from conjugate_gradient_functions import nonlinear_conjugate_gradient
        from myClasses import Dispersion
        import positions

        generation = 'RSPIRAL'  # SPIRAL, RSPIRAL, CHESSBOARD, RANDOM

        L, H = self.rveSize
        Dispersion.resetList()
        dispersion = Dispersion(L=L, H=H, N=0)

        ### Add existing particles first (add fibres case)
        for fiber in self.sq:
            if fiber.period in [0,1,2,5]:
                fixed = False
                if fiber.period==5:  # Only fix corner fiber
                    fixed = True
                x0, y0 = fiber.center
                dispersion.setParticle(L=fiber.L + self.tolerance,
                                       x0=x0, y0=y0, fixed=fixed,
                                       phi=fiber.phi,
                                       shape=fiber.geometry, parameters=fiber.parameters,
                                       material=fiber.material)
                dispersion.N += 1

        ### Particles generation: add fibersets sequentially
        for i, f_set in enumerate(fibersets):
            # Add the fibers corresponding to each fiberset
            localVf = 0.0
            n = 0  # Number of fibers in this fiberset
            shape = f_set['Geometry']
            params = f_set['Parameters']
            df, d_df = f_set['df']  # Equivalent diameter
            phi, d_phi = f_set['Phi']
            factor = fiber_shape(shape, params, 1.0)
            Lf = df * factor  # Circumscribing diameter

            # Precompute size and number of fibres
            Lf_list = []
            while localVf < f_set['Vf']/100:
                # Generate random size (d)
                cur_df = generateRandomNormal(df, d_df)
                Lf_list.append(cur_df * factor)
                localVf += 0.25*np.pi*cur_df*cur_df/(L*H)

            # Generate fibre center coordinates
            N = len(Lf_list)
            centers = []

            if generation.upper() == 'CHESSBOARD':
                # Chessboard-like generation
                tx = int(L/df)
                dx = L/tx
                ty = int(H/df)
                dy = L/ty
                # print 'Chessboard', dx, dy, tx, ty, N
                tiles = sum([[(i,j) for i in range(tx)] for j in range(ty)], [])
                current_tiles = [t for t in tiles]
                random.shuffle(current_tiles)
                for i in range(N):
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
                    centers.append((x0,y0))
                    if not current_tiles:
                        current_tiles = [t for t in tiles]
                        random.shuffle(current_tiles)

            elif generation.upper()=='SPIRAL':
                # Archimedean spiral
                if L>H:
                    dx = df
                    dy = H/L*df
                else:
                    dy = df
                    dx = L/H*df

                a = 0.5*df/np.pi
                tmax = max(L*0.5, H*0.5)/a

                # Create spiral
                spiral = positions.archimedeanSpiral(a, tmax)
                # Generate equally spaced points
                newTheta = spiral.generatePoints(N=N)
                r, theta, x, y = spiral.angleToPoints(newTheta)
                # Translate (x_c, y_c), scale (L/H) and apply random noisy displacement
                x0 = 0.5*L + dx/df * (x + dx*(2*np.random.random(N)-1))
                y0 = 0.5*H + dy/df * (y + dy*(2*np.random.random(N)-1))
                centers += zip(x0,y0)

            elif generation.upper()=='RSPIRAL':
                # Rectangular spiral
                if L>H:
                    dx = df
                    dy = H/L*df
                else:
                    dy = df
                    dx = L/H*df

                if dx>dy:
                    n = int((L-dx)/dx/2)
                else:
                    n = int((H-dy)/dy/2)

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
                x, y = zip(*points)
                # Translate (x_c, y_c), scale (L/H) and apply random noisy displacement
                x0 = np.array(x) + 0.5*L + df*(2.5*np.random.random(N)-1.25)
                y0 = np.array(y) + 0.5*H + df*(2.5*np.random.random(N)-1.25)
                centers += zip(x0,y0)

            else:
                # Random generation
                for i in range(N):
                    x0 = 0.5*Lf + (L-Lf)*np.random.rand()
                    y0 = 0.5*Lf + (H-Lf)*np.random.rand()
                    centers.append((x0,y0))


            print len(centers), N, len(Lf_list)

            for i, lf in enumerate(Lf_list):
                if dispersion.N == 0: # First fiber is fixed at the origin (corner fiber)
                    x0, y0 = 0.0, 0.0
                    fixed = True
                else:
                    x0, y0 = centers[i]
                    fixed = False

                # Create particle
                # Apply tolerance distance between fibres (increase size L fictionally)
                dispersion.setParticle(L=lf+self.tolerance,
                                      x0=x0, y0=y0, fixed=fixed,
                                      phi= phi + d_phi*(-1 + 2*np.random.random()), #generateRandomNormal(phi, d_phi),
                                      shape=shape, parameters=params,
                                      material=f_set['Material'])
                n += 1
                dispersion.N += 1
                # localVf += 0.25*np.pi*cur_df*cur_df/(L*H)

            print 'Fiberset {0:d}. Fibers {1:d}, V = {2:.1f}%'.format(i+1, n, localVf*100.)

        # Apply tolerance distance between fibres (increase size fictionally)
        # [p.set_L(increment=self.tolerance) for p in dispersion.Particles.values()]

        # Dispersion update
        dispersion.update()
        # print dispersion
        if plot: dispersion.plot(title=True, numbering=False, ion=True)

        ### SOLVER: Potential application
        # t0 = time.time()
        nonlinear_conjugate_gradient(d_potential, potential, dispersion, 1.0e-6, plot=plot, verbose=verbose, maxiter=1000)
        # print 'Time: {0:.1f} s'.format(time.time()-t0)

        # Correct tolerance distance between fibres (decrease size fictionally)
        [p.set_L(increment=-self.tolerance) for p in dispersion.Particles.values()]

        if plot:
            print dispersion
            dispersion.plot(title=True, numbering=True, margin=0.0, arrows=False, ion=True)

        # Translate dispersion of particles into fibers distribution
        Fiber.resetList()
        self.sq = []
        for k, p in dispersion.Particles.items():
            f = Fiber(geometry=p.shape, parameters=p.parameters, L=p.L, phi=p.phi, center=p.center,
                      period=0, Nf=1, material=p.material)
            self.sq.append(f)

        # Double-check final layout
        self.validate()
        self.setPeriodicity()

        return 0

    def wrapDynamicGeneration(self, fibersets, optPeriod):

        L, H = self.rveSize

        ### Particles generation: add fibersets sequentially
        for i, f_set in enumerate(fibersets):
            # Add the fibers corresponding to each fiberset
            localVf = 0.0
            n = 0  # Number of fibers in this fiberset
            shape = f_set['Geometry']
            params = f_set['Parameters']
            df, d_df = f_set['df']  # Equivalent diameter
            phi, d_phi = f_set['Phi']
            factor = fiber_shape(shape, params, 1.0)
            Lf = df * factor  # Circumscribing diameter
            d_Lf = d_df * factor

            # Compute size and number of fibres
            while localVf < f_set['Vf']/100:
                if not bool(self.sq) and (optPeriod == FULL):
                    cur_df = self.origin_fiber(shape, params, f_set['Material'], (Lf, d_Lf), f_set['Phi'])
                    localVf += 0.25*np.pi*cur_df*cur_df/(L*H)
                else:
                    # Generate random fiber size (df)
                    cur_df = generateRandomNormal(df, d_df)
                    # Generate random position
                    x0 = L*np.random.rand()
                    y0 = H*np.random.rand()
                    # Check center is not contained into other fiber
                    if self._inside(x0, y0): continue

                    # Generate fiber
                    fiber = Fiber(geometry=shape, parameters=params, material=f_set['Material'], L=cur_df*factor,
                              phi=phi + d_phi*(-1 + 2*np.random.random()), center=(x0,y0), Nf=1)
                    # Check if fiber is periodic and it is allowed
                    p = self.periodicity_check(fiber)
                    fiber.set_period(p)
                    if p != 0:
                        if p in periodicity[optPeriod]+[5,6,7,8]: continue
                        self.sq.append(fiber)
                        self.periodic_fiber(fiber)
                        valid = self.valid_fiber(fiber, optPeriod)
                        if not valid:
                            del self.sq[-2:]
                            continue

                        # print fiber.period, self.sq[-1].period
                    else:
                        self.sq.append(fiber)

                    localVf += 0.25*np.pi*cur_df*cur_df/(L*H)
                n += 1


            print 'Fiberset {0:d}. Fibers {1:d}, V = {2:.1f}%'.format(i+1, n, localVf*100.)

        # self.plot_rve(show_plot=True, numbering=True)

        # Apply algorithm
        generateBox2D(self, margin=0.0, lmax=600.0, gravity=(0,0), autoStop=True, crack=None, tolerance=self.tolerance)

        return 0

    def pixelGrid(self, pixelSize, spots=True, fibres=True, plot=True, save='pixelgrid.txt'):

        L, H = self.rveSize
        # Mx = int(np.ceil(spotSize/L))
        # My = int(np.ceil(spotSize/H))

        # Make an array Mx x My
        # pmicro = patternedMicrostructure(self, ncols=2*Mx+1, nrows=2*My+1, plot=False, save=False)

        # Create grid (cells)
        dx = dy = pixelSize
        nx = np.ceil(L/dx)+1
        xv = np.linspace(0, L, nx)
        ny = np.ceil(H/dy)+1
        yv = np.linspace(0, H, ny)
        # nx = np.ceil(L/dx)
        # xv = np.linspace(Mx*L+dx/2, (Mx+1)*L-dx/2, nx)
        # ny = np.ceil(H/dy)
        # yv = np.linspace(My*H+dy/2, (My+1)*H-dy/2, ny)
        print nx, ny
        xx, yy = np.meshgrid(xv, yv)
        # Check pixel by pixel if contained in a fibre
        Chi = np.array([ [ 2-self._inside(x, y) for x in xv ] for y in yv ])
        #Chi = np.array([ [ self._inside(x, y) for x in xv ] for y in yv ])

        if save:
            f = open(save, 'w')
            for row in Chi:
                #f.write(str(row)[1:-1]+'\n')
                for v in row:
                    f.write(' {0:d}'.format(v))
                f.write('\n')
            f.close()

        if plot:
            fig, ax0 = plt.subplots(facecolor='w', edgecolor='k')
            im = ax0.contourf(xx, yy, Chi)
            fig.colorbar(im, ax=ax0)

            # Plot computation points
            if spots: ax0.plot(xx, yy, '.', ms=2, color='k')

            # Plot fibres
            if fibres:
                for i, fiber in enumerate(self.sq):
                    polyvert = np.asarray(fiber.polygonly.exterior)
                    fiberpoly = mplPolygon(polyvert, facecolor='w', edgecolor='k', fill=False, lw=2)
                    ax0.add_patch(fiberpoly)

            # ax0.set_xlim(Mx*L, (Mx+1)*L)
            # ax0.set_ylim(My*H, (My+1)*H)
            ax0.set_aspect('equal')
            plt.show()
            ax0.set_xticks([])
            ax0.set_yticks([])
            # plt.ion()



        return xx, yy, Chi

    def pixelGrid2(self, nx, spots=True, fibres=True, plot=True, save='pixelgrid.txt', javi=False):
        """
        Efficient raster of the microstructure through image processing (PIL)
        1: Fibre
        2: Matrix
        :param nx: number of points along the x direction
        :param spots: plot spots
        :param fibres: plot fibres contour
        :param plot:
        :param save: filename to save the matrix-like raster
        :return:
        """

        L, H = self.rveSize

        # Create grid (cells)
        dx = L/nx
        dy = dx
        ny = int(np.ceil(H/dy))
        print dx, dy
        print nx, ny

        # Create auxiliary image to read from pixel colour
        dpi = nx
        self.easy_plot(filename='simple_plot2', show_plot=False, lmax=10., dpi=dpi)

        img = Image.open('simple_plot2.png')
        pixels = img.load()
        sx, sy = img.size
        # print sx,sy
        dsx = float((sx-4.)/nx)  # Neglect 2 starting and 2 ending pixels
        dsy = float((sy-4.)/ny)  # Neglect 2 starting and 2 ending pixels
        # print dsx, dsy

        # Check pixel by pixel if contained in a fibre. 0: white, 255: black
        # Chi = np.array([ [ int(pixels[2+x*dsx, sy-2-y*dsy][0]/255)+1 for x in range(0,nx) ]  for y in range(0,ny)])
        Chi = np.array([ [ pixels[2+x*dsx, sy-2-y*dsy][0] for x in range(0,nx) ]  for y in range(0,ny)])
        if javi: Chi = Chi/255+1
        else: Chi = -Chi/255+1

        if save:
            f = open(save, 'w')
            for row in Chi:
                #f.write(str(row)[1:-1]+'\n')
                for v in row:
                    f.write(' {0:d}'.format(v))
                f.write('\n')
            f.close()

        xv = np.linspace(0, L, nx)
        yv = np.linspace(0, H, ny)
        xx, yy = np.meshgrid(xv, yv)
        if plot:
            fig, ax0 = plt.subplots(facecolor='w', edgecolor='k')
            im = ax0.contourf(xx, yy, Chi)
            fig.colorbar(im, ax=ax0)

            # Plot computation points
            if spots: ax0.plot(xx, yy, '.', ms=2, color='k')

            # Plot fibres
            if fibres:
                for i, fiber in enumerate(self.sq):
                    polyvert = np.asarray(fiber.polygonly.exterior)
                    fiberpoly = mplPolygon(polyvert, facecolor='w', edgecolor='k', fill=False, lw=2)
                    ax0.add_patch(fiberpoly)

            # ax0.set_xlim(Mx*L, (Mx+1)*L)
            # ax0.set_ylim(My*H, (My+1)*H)
            ax0.set_aspect('equal')
            plt.show()
            ax0.set_xticks([])
            ax0.set_yticks([])
            # plt.ion()

        return xx, yy, Chi

    def easy_plot(self, filename='simple_plot', directory='', imageFormat='png', show_plot=True, matrix_colour='w',
                  fibre_colour='k', dpi=100, lmax=10.):
        """
        Function: easy_plot
        Easy plot RVE cell layout.\n
        Requires matplotlib package in PYTHONPATH
        """
        try:
            from matplotlib.patches import Polygon as mplPolygon
            from matplotlib.patches import Rectangle
        except ImportError:
            raise ImportError, 'Cannot plot microstructure'

        if not bool(directory):
            directory = '.'

        maxDimension = float(max(self.rveSize))
        sizex = self.rveSize[0] / maxDimension * lmax
        sizey = self.rveSize[1] / maxDimension * lmax
        fig, ax = plt.subplots(num=None, figsize=(sizex, sizey), dpi=dpi, facecolor='w', edgecolor='w')

        L, H = self.rveSize
        ax.add_patch(Rectangle((0, 0), L, H, facecolor=matrix_colour))
        ax.set_xlim((0, L))
        ax.set_ylim((0, H))
        ax.set_aspect('equal')
        ax.set_xticks([0, L])
        ax.set_yticks([0, H])

        for i, fiber in enumerate(self.sq):

            # fiberpoly  = self.sq[i].poly
            if fiber.geometry.upper()==CIRCULAR:
                fiberpoly = mplCircle(fiber.center, radius=fiber.L*0.5, facecolor=fibre_colour, edgecolor=fibre_colour)
            else:
                polyvert = np.asarray(fiber.polygonly.exterior)
                fiberpoly = mplPolygon(polyvert, facecolor=fibre_colour, edgecolor=fibre_colour)

            ax.add_patch(fiberpoly)

        ax.axis('off')
        ax.set_title('')
        ax.set_xticks([])
        ax.set_yticks([])

        #plt.tight_layout()
        # Adjust figure to delete margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if filename:
            fig.savefig(os.path.join(directory, '.'.join([filename, imageFormat])))

        # plt.tight_layout()
        if show_plot:
            plt.ioff()
            plt.show()
        else:
            # Clear workspace
            plt.close()

        return fig, ax

    def S2_function(self, pixelSize, spots=True, fibres=True, plot=True):

        # L, H = self.rveSize
        xx, yy, Chi = self.pixelGrid(pixelSize=pixelSize, spots=spots, fibres=fibres, plot=False)

        # Compute S_2 function as periodic medium:
        # Jan Havalka, Ana Kucerova, Jan Sykora, "Compression and reconstruction of random microstructures
        # using accelerated lineal path function", Computational Materials Science, 2016

        DFT = np.fft.fft2(Chi)
        S_2 = np.real(np.fft.ifft2(DFT * np.matrix.conjugate(DFT)) / (len(xx)*len(yy)))
        # interp_S_2 = interpolate.bisplrep(xx, yy, S_2, s=0)
        interp_S_2 = interpolate.interp2d(xx, yy, S_2, kind='cubic')

        # Interpolate for constant angles
        dmax = min(self.rveSize)
        r = np.linspace(0,dmax,100)
        angles = np.linspace(0, 90., 5)
        curves_phi = []
        for angle in angles:
            phi = angle*np.pi/180.
            x_ = np.cos(phi)*r
            y_ = np.sin(phi)*r
            S2_ = interp_S_2(x_, y_)
            S2_ = [S2_[i][i] for i in range(len(S2_))]
            curves_phi.append([x_, y_, S2_, angle])

        # Interpolate for constant distances
        distances = np.linspace(0, dmax, 5)
        curves_d = []
        Lphi = np.linspace(0, np.pi/2, 100)
        for d in distances:
            x_ = np.cos(Lphi)*d
            y_ = np.sin(Lphi)*d
            S2_ = [interp_S_2(xnew, ynew)[0] for xnew, ynew in zip(x_,y_)]
            # S2_ = [S2_[i][i] for i in range(len(S2_))]
            curves_d.append([x_, y_, S2_, d])

        if plot:
            fig = plt.figure(facecolor='w', edgecolor='k', figsize=(12,12))

            #---- First subplot
            ax0 = fig.add_subplot(2, 2, 1, projection='3d')
            surf = ax0.plot_surface(xx, yy, S_2, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.3)

            # Plot: phi = cte
            for curve in curves_phi:
                ax0.plot(curve[0], curve[1], curve[2], label=r'$\phi=${0:.2f}'.format(curve[3])+'$^{\circ}$')

            # Plot: d = cte
            for curve in curves_d:
                ax0.plot(curve[0], curve[1], curve[2], ls='--', label=r'$d=${0:.2f}'.format(curve[3])+'$\mu m$')

            fig.colorbar(surf, shrink=0.5, aspect=10)

            # fig, ax = plt.subplots(facecolor='w', edgecolor='k', ncols=2, figsize=(12,6))
            im = ax0.contourf(xx, yy, S_2, zdir='z', offset=0)
            # fig.colorbar(im, ax=ax[0])
            # ax0.set_aspect('equal')
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_zlim(0, max([max(s) for s in S_2]))
            #

            #---- Second subplot
            ax1 = fig.add_subplot(2, 2, 3)
            ax1.contourf(xx, yy, np.real(Chi))

            # Plot computation points
            if spots: ax0.plot(xx, yy, '.', ms=2, color='k')

            # Plot fibres
            import mpl_toolkits.mplot3d.art3d as art3d
            if fibres:
                for i, fiber in enumerate(self.sq):
                    polyvert = np.asarray(fiber.polygonly.exterior)
                    fiberpoly = mplPolygon(polyvert, facecolor='w', edgecolor='k', fill=False, lw=2)
                    ax0.add_patch(fiberpoly)
                    art3d.pathpatch_2d_to_3d(fiberpoly, z=0, zdir="z")

            ax1.set_aspect('equal')
            #

            #---- Third subplot
            ax2 = fig.add_subplot(2, 2, 2)

            # Interpolated phi
            for curve in curves_phi:
                ax2.plot(r, curve[2], label=r'$\phi=${0:.2f}'.format(curve[3])+'$^{\circ}$')
            leg = ax2.legend()
            leg.draggable()
            ax2.set_ylim([0, 1.1*max([max(s) for s in S_2])])
            ax2.set_xlabel(r'$d\, (\mu m)$', fontsize=20)
            ax2.set_ylabel(r'$S_2\, (\cdot)$', fontsize=20)
            ax2.set_title(r'Orientation arrangement analysis', fontsize=20)
            formatTicks(ax2, fontsize=18, x=True, y=True, z=False, color='black')
            #

            #---- Fourth subplot
            ax4 = fig.add_subplot(2, 2, 4)

            # Interpolated d
            for curve in curves_d:
                ax4.plot(Lphi*180./np.pi, curve[2], ls='--', label=r'$d=${0:.2f}'.format(curve[3])+'$\mu m$')
            leg = ax4.legend()
            leg.draggable()
            ax4.set_ylim([0, 1.1*max([max(s) for s in S_2])])
            ax4.set_xlabel(r'$\phi\, (^{\circ})$', fontsize=20)
            ax4.set_ylabel(r'$S_2\, (\cdot)$', fontsize=20)
            ax4.set_title(r'Distance arrangement analysis', fontsize=20)
            formatTicks(ax4, fontsize=18, x=True, y=True, z=False, color='black')
            #

            plt.show()
            # plt.ion()

        return xx, yy, S_2

    def _inside(self, x, y):
        """
        Check if point (x,y) falls within a fibre
        :param x: x-coordinate, float
        :param y: y coordinate, float
        :return: 1 if point (x,y) falls within a fibre
        """

        point = Point(x,y)

        # Filter close enough fibres
        candidates = [f for f in self.sq if dist2points((x,y), f.center)<f.L*0.5]

        return int(bool([1 for f in candidates if f.polygonly.contains(point)]))

    def voronoi(self, pattern=True, plot=True, plotFibers=True):

        from scipy.spatial import Voronoi
        L, H = self.rveSize

        if pattern:
            auxMicro = patternedMicrostructure(self, ncols=3, nrows=3, plot=False)
            points = np.array([f.center for f in auxMicro.sq])
            l0, h0 = L, H
            l, h = 2*L, 2*H
            lm, hm = 3*L, 3*H
        else:
            points = np.array([f.center for f in self.sq])
            l0, h0 = 0, 0
            l, h = L, H
            lm, hm= L, H

        vor = Voronoi(points)

        # Interpret Voronoi results
        vertices = vor.vertices
        polygons = []
        for r in vor.regions:
            if (-1 not in r) and (len(r)>0):


                # Check that at least one vertex is within l0<x<l and h0<y<h
                countRegion1 = False
                for v in [vertices[i] for i in r]:
                    if l0<v[0]<l and h0<v[1]<h:
                        countRegion1 = True
                        break

                if not countRegion1: continue

                # Check that all vertices are within whole domain
                countRegion2 = True
                for v in [vertices[i] for i in r]:
                    if not (0<v[0]<lm and 0<v[1]<hm):
                        countRegion2 = False
                        break

                if not countRegion2: continue

                # print r, countRegion1, countRegion2
                # if not(countRegion1 or countRegion2): continue

                # Check if any vertices lie in periodicity 2 (y>h) or 3 (x>l)
                countRegion3 = True
                for v in [vertices[i] for i in r]:
                    if v[0]>l or v[1]>h:
                        countRegion3 = False
                        break

                if not countRegion3: continue

                # Compute area of the region 'r'
                poly = Polygon([vertices[i] for i in r])
                polygons.append(poly)
                # print r, poly.area

        areas = [poly.area for poly in polygons]
        std = np.std(areas)
        mean = np.mean(areas)

        # Default plot
        # from scipy.spatial import voronoi_plot_2d
        # voronoi_plot_2d(vor)

        # Custom plot
        if plot or plotFibers:
            from matplotlib.patches import Rectangle as mplRectangle
            from matplotlib.patches import Polygon as mplPolygon
            m_area = max(areas)*1.05
            maxDimension = float(max(self.rveSize))
            lmax = 6.0
            sizex = self.rveSize[0] / maxDimension * lmax
            sizey = self.rveSize[1] / maxDimension * lmax
            fig, ax = plt.subplots(num=None, figsize=(sizex, sizey), dpi=100, facecolor='w', edgecolor='k')
            #
            ax.set_xlim((0, L))
            ax.set_ylim((0, H))
            ax.set_aspect('equal')
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.set_xticks([0, L])
            ax.set_yticks([0, H])

            for i, poly in enumerate(polygons):
                polyvert = np.asarray(poly.exterior) - np.array([l0,h0])
                # polyvert = np.asarray(poly.exterior)
                fiberpoly = mplPolygon(polyvert, facecolor='red', alpha=1-poly.area/m_area, edgecolor='k', lw=2.)
                # fiberpoly = mplPolygon(polyvert, facecolor='red', edgecolor='k', lw=2.)
                ax.add_patch(fiberpoly)
                ax.plot(poly.centroid.x-L, poly.centroid.y-H, 'xk', ms=4, mew=1.)

            if plotFibers:
                for i, fiber in enumerate(self.sq):
                    if fiber.geometry.upper()==CIRCULAR:
                        fiberpoly = mplCircle(fiber.center, radius=fiber.L*0.5, facecolor='None', edgecolor='k')
                        ax.add_patch(fiberpoly)

            ax.add_patch(mplRectangle((0, 0), L, H, facecolor='y', edgecolor='k', ls='dashed', alpha=0.2))

            sMean = '{0:.1f}'.format(mean)
            sStd = '{0:.1f}'.format(std)
            sRatio = '{0:.3f}'.format(std/mean)
            ax.set_title(r'$\mu_v='+sMean+',\, \sigma_v='+sStd+',\, \sigma_v/\mu_v='+sRatio+'$')
            # ax.set_title(r'$\mu_v=$'+'{0:.1f}'.format(mean)+', $\sigma_v=$'+'{0:.1f}'.format(std))

            plt.show()

        return mean, std


class Fiber:
    """Fiber class\n
    Attributes:\n
    - shape (shape). square, circle\n
    - size (L). Characteristic size of the fiber section\n
    - Orientation in degrees (phi). Angle respect the horizontal axis\n
    - Center (x,y). Position on the xy plane of the midpoint of the fiber section\n
    - Periodicity (P). Determination of the periodicity. 0 is non-periodic\n
    - Symmetricity (Nf). \n\n  
    Methods:\n
    + coord2vert. Compute vertices based on its attributes\n
    + Poly. Object of class  patch of matplotlib\n
    """
    List = list()
    Id = 0  # Id count

    def __init__(self, geometry, parameters=[], material='CF-AS4', L=1.0, phi=0.0, center=(0.0, 0.0), period=0,
                 Nf=1, aux=False):
        """
        Fiber class
        """
        self.id = None
        self.geometry = geometry
        self.parameters = parameters    # default []
        self.material = material        # default 'CF-AS4'
        self.L = L                      # default L = 1
        self.phi = phi                  # default phi = 0
        self.center = center            # default (0,0)
        self.period = period            # default period = 0
        self.Nf = Nf                    # default Nf = 1
        self.vertices = coord2vert(geometry=geometry, L=L, center=center, phi=phi, parameters=parameters)
        self.polygonly = self.set_polygonly()
        if not aux:
            Fiber.List.append(self)
            self.poly = self.set_poly()
            self.id = Fiber.Id
            Fiber.Id += 1

    def set_material(self, material):
        self.material = material
        fibercolour, _ = colourFiber(self.material)
        self.poly.set_facecolor(fibercolour)

    def set_shape(self, geometry, parameters):
        self.geometry = geometry
        self.parameters = parameters
        self.vertices = coord2vert(geometry=self.geometry, L=self.L, center=self.center, phi=self.phi, parameters=self.parameters)
        self.polygonly = self.set_polygonly()
        self.poly = self.set_poly()

    def set_size(self, L):
        self.L = L
        self.vertices = coord2vert(geometry=self.geometry, L=self.L, center=self.center, phi=self.phi, parameters=self.parameters)
        self.polygonly = self.set_polygonly()
        self.poly = self.set_poly()

    def set_phi(self, phi):
        # Optimize to produce rotation simply
        #        deltaphi = phi-self.phi
        self.phi = phi
        self.vertices = coord2vert(geometry=self.geometry, L=self.L, center=self.center, phi=self.phi, parameters=self.parameters)
        self.polygonly = self.set_polygonly()
        self.poly = self.set_poly()

    def set_center(self, center):
        # Optimize to produce translation simply
        deltaU = (center[0] - self.center[0], center[1] - self.center[1])
        self.center = center
        self.vertices[0] += deltaU[0]
        self.vertices[1] += deltaU[1]
        self.polygonly = self.set_polygonly()
        self.poly = self.set_poly()

    def set_period(self, period):
        self.period = period

    def set_Nf(self, Nf):
        self.Nf = Nf

    def __str__(self):
        return 'shape = %s\nL = %6.3f\nphi = %4.2f\ncenter = (%4.2f,%4.2f)\nperiod = %g\nNf = %g' % (
            self.geometry, self.L, self.phi, self.center[0], self.center[1], self.period, self.Nf)

    def set_polygonly(self):
        """Returns a shapely Polygon object"""
        return Polygon(zip(self.vertices[0], self.vertices[1]))

    def set_poly(self):
        polyvert = np.asarray(self.polygonly.exterior)
        fibercolour, _, _ = colourFiber(self.material)
        return mplPolygon(polyvert, facecolor=fibercolour, edgecolor='black')

    def destroy(self):
        Fiber.List.remove(self)

    @staticmethod
    def resetList():
        Fiber.List = []
        Fiber.Id = 0

############################################################################################
# Box2D functions

def my_draw_polygon(polygon, body, PPM, SCREEN_HEIGHT_PX, screen):
    vertices=[(body.transform*v)*PPM for v in polygon.vertices]
    vertices=[(v[0], SCREEN_HEIGHT_PX-v[1]) for v in vertices]
    material = body.userData.material  # Fiber material
    fibercolour, textcolour, edgecolour = colourFiber(material)
    pygame.draw.polygon(screen, colors[fibercolour], vertices, 0)
b2PolygonShape.draw = my_draw_polygon

def my_draw_circle(circle, body, PPM, SCREEN_HEIGHT_PX, screen):
    position=body.transform*circle.pos*PPM
    position=(position[0], SCREEN_HEIGHT_PX-position[1])
    material = body.userData.material
    fibercolour, textcolour, edgecolour = colourFiber(material)
    pygame.draw.circle(screen, colors[fibercolour], [int(x) for x in position], int(circle.radius*PPM))
b2CircleShape.draw = my_draw_circle

def my_draw_edge(edge, body, PPM, SCREEN_HEIGHT_PX, screen):
    vertices=[(body.transform*v)*PPM for v in edge.vertices]
    vertices=[(v[0], SCREEN_HEIGHT_PX-v[1]) for v in vertices]
    # vertices = fix_vertices([body.transform*edge.vertex1*PPM, body.transform*edge.vertex2*PPM])
    pygame.draw.line(screen, (0,255,64), vertices[0], vertices[1])
b2EdgeShape.draw = my_draw_edge

def createBody(world, fiber, center, periodicSettings, tol=0., dynamic=True, periodic=0):
    if tol:
        fiber.set_size(fiber.L + tol)

    if dynamic:
        body = world.CreateDynamicBody(position=center,
                                       # restitution=1.0,
                                       # density=1.0,
                                       userData=fiber)
    else:
        body = world.CreateStaticBody(position=center,
                                      # restitution=1.0,
                                      # density=1.0,
                                      userData=fiber)

    # Polygonal fibres
    if (fiber.geometry.upper() == POLYGON) and (len(fiber.vertices[0]) < b2_maxPolygonVertices):
        new_x = fiber.vertices[0] - fiber.center[0]
        new_y = fiber.vertices[1] - fiber.center[1]
        verts = zip(new_x, new_y)[-1:0:-1]
        myShape = b2PolygonShape(vertices=verts)
        body.CreateFixture(shape=myShape, density=1, friction=0.1)
        if periodic != 0:
            new_x += periodicSettings[periodic]['dx']
            new_y += periodicSettings[periodic]['dy']
            verts = zip(new_x, new_y)[-1:0:-1]
            myShape = b2PolygonShape(vertices=verts)
            body.CreateFixture(shape=myShape, density=1, friction=0.1)
    # Circular fibres
    elif fiber.geometry.upper() == CIRCULAR:
        myShape = b2CircleShape(radius=fiber.L * 0.5)
        body.CreateFixture(shape=myShape, density=1, friction=0.1)
        if periodic != 0:
            centerP = (periodicSettings[periodic]['dx'],
                       periodicSettings[periodic]['dy'])
            myShape = b2CircleShape(radius=fiber.L * 0.5, pos=centerP)
            body.CreateFixture(shape=myShape, density=1, friction=0.1)
    # Vertices-based fibres
    else:
        new_x = fiber.vertices[0] - fiber.center[0]
        new_y = fiber.vertices[1] - fiber.center[1]
        verts = zip(new_x, new_y)[-1::-1]
        for i in range(1, len(verts)):
            # myShape = b2CircleShape(center=verts[i], radius=fiber.L*0.15)
            myShape = b2PolygonShape(vertices=[verts[i - 1], verts[i], (0, 0)])
            body.CreateFixture(shape=myShape, density=1)
        myShape = b2PolygonShape(vertices=[verts[i], verts[0], (0, 0)])
        body.CreateFixture(shape=myShape, density=1, friction=0.1)
        if periodic != 0:
            offset = (periodicSettings[periodic]['dx'],
                      periodicSettings[periodic]['dy'])
            new_x += offset[0]
            new_y += offset[1]
            verts = zip(new_x, new_y)[-1::-1]
            for i in range(1, len(verts)):
                myShape = b2PolygonShape(vertices=[verts[i - 1], verts[i], offset])
                body.CreateFixture(shape=myShape, density=1)
            myShape = b2PolygonShape(vertices=[verts[i], verts[0], offset])
            body.CreateFixture(shape=myShape, density=1, friction=0.1)

    # else:
    #     raise NotImplementedError('Fibre shape {} not implemented yet'.format(fiber.geometry))

    return body

def generateBox2D(micro, margin=0.0, lmax=600.0, gravity=(0,0), autoStop=True, crack=None, tolerance=0.1,
                  plot=False, verbose=False):

    maxDimension = float(max(micro.rveSize))
    PPM = lmax / maxDimension  # pixels per unit length
    # print 'PPM = ', PPM

    w, h = micro.rveSize
    RVE_WIDTH_PX = round(w * PPM)
    RVE_HEIGHT_PX = round(h * PPM)
    SCREEN_OFFSETX_PX, SCREEN_OFFSETY_PX = margin, margin

    SCREEN_WIDTH_PX = int(RVE_WIDTH_PX + 2 * SCREEN_OFFSETX_PX)
    SCREEN_HEIGHT_PX = int(RVE_HEIGHT_PX + 2 * SCREEN_OFFSETY_PX)

    # print 'Screen (pixels): %d x %d' % (SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
    # print 'RVE (pixels): %d x %d' % (RVE_WIDTH_PX, RVE_HEIGHT_PX)

    W = SCREEN_WIDTH_PX / PPM
    H = SCREEN_HEIGHT_PX / PPM
    margin_x, margin_y = SCREEN_OFFSETX_PX / PPM, SCREEN_OFFSETY_PX / PPM

    # Handler for periodic fibres
    periodicSettings = {
        # Parameters to generate the periodic fiber of a p1-fiber (p3)
        1: {'dx': w, 'dy': 0.0, 'limit': h,  # 'localAnchor': (W,0),
            'moveAxis': (0, 1), 'constraintAxis': (1, 0), 'margin': margin_y},

        # Parameters to generate the periodic fiber of a p4-fiber (p2)
        4: {'dx': 0.0, 'dy': h, 'limit': w,  # 'localAnchor': (0,H),
            'moveAxis': (1, 0), 'constraintAxis': (0, 1), 'margin': margin_x},
    }

    # --- pygame setup ---
    screen = pygame.display.set_mode((SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX), 0, 32)
    pygame.display.set_caption('Physical compaction')
    clock = pygame.time.Clock()

    # --- pybox2d world setup ---
    # Create the world
    world = b2World(gravity=gravity, doSleep=True)
    # world = b2World(doSleep=True)

    # Static edges to hold the fibers inside
    shapes = [
        b2EdgeShape(vertices=[(margin_x + tolerance, margin_y), (margin_x + tolerance, H - margin_y)]),  # Left edge
        b2EdgeShape(vertices=[(margin_x, H - margin_y - tolerance), (W - margin_x, H - margin_y - tolerance)]),
        # Top edge
        b2EdgeShape(vertices=[(W - margin_x - tolerance, H - margin_y), (W - margin_x - tolerance, margin_y)]),
        # Right edge
        b2EdgeShape(vertices=[(W - margin_x, margin_y + tolerance), (margin_x, margin_y + tolerance)]),  # Bottom edge
    ]

    rve = world.CreateStaticBody(shapes=shapes)

    fiberBodies = list()
    numberOfDynamicBodies = 0
    myPrevPos = []
    for fiber in micro.sq:
        center = (fiber.center[0] + margin_x, fiber.center[1] + margin_y)
        # minDist = tolerance
        # print 'Time for distance: %.2f' % (time.time()-t1)
        p = fiber.period

        if p == 0:
            body = createBody(world, fiber, center, periodicSettings, tolerance)
            numberOfDynamicBodies += 1
            fiberBodies.append(body)
            myPrevPos.append(body.position.length)

        elif p in [5, 6, 7, 8]:
            body = createBody(world, fiber, center, periodicSettings, tolerance, dynamic=False)
            fiberBodies.append(body)
            myPrevPos.append(body.position.length)

        elif p in [1, 4]:
            body = createBody(world, fiber, center, periodicSettings, tolerance, periodic=p)
            numberOfDynamicBodies += 1
            fiberBodies.append(body)
            myPrevPos.append(body.position.length)

            # Implement periodic constraint
            if p == 1:
                c = (center[0], 0)
            else:
                # p = 4
                c = (0, center[1])

            # Fiber - RVE
            world.CreatePrismaticJoint(
                bodyA=rve,
                bodyB=body,
                enableLimit=False,
                localAnchorA=c,
                localAnchorB=(0, 0),
                axis=periodicSettings[p]['moveAxis'],
                # lowerTranslation=-h,
                # upperTranslation=h,
            )

    # --- main game loop ---
    nbodies = len(fiberBodies)
    running = True
    time0 = time.time()
    time2stop = 0.1
    velAverageRecord = list()
    velMaxRecord = list()
    timeRecord = list()

    # micro.save_rve(filename='pre.txt')
    inc = 0
    while running:
        # Check the event queue
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                running = False

        screen.fill(colors['background'])

        # Draw the fibers
        for body in fiberBodies:
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                # velocity += body.linearVelocity.length
                # try:
                    fixture.shape.draw(body, PPM, SCREEN_HEIGHT_PX, screen)
                # except:
                #     print 'Could not draw body'

        if (time.time() - time0 >= time2stop) and autoStop:
            velocities = [abs(fiberBodies[i].position.length - myPrevPos[i]) / TIME_STEP for i in range(nbodies)]
            # velAverage = sum([body.linearVelocity.length for body in fiberBodies])/numberOfDynamicBodies
            # velMax = max([body.linearVelocity.length for body in fiberBodies])
            velAverage = sum(velocities) / numberOfDynamicBodies
            velMax = max(velocities)
            # print 'Average velocity %f' % velocity
            velAverageRecord.append(velAverage)
            velMaxRecord.append(velMax)
            timeRecord.append(time2stop)
            myPrevPos = [body.position.length for body in fiberBodies]
            if velMax < 0.05 and inc > 0:  # or (velocity < max(velocityRecord)/50.0):
                running = False
            time2stop += TIME_STEP

        # Draw the RVE boundaries
        for body in (rve,):
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                fixture.shape.draw(body, PPM, SCREEN_HEIGHT_PX, screen)

        # Make Box2D simulate the physics of our world for one step.
        # Instruct the world to perform a single step of simulation. It is
        # generally best to keep the time step and iterations fixed.
        # See the manual (Section "Simulating the World") for further discussion
        # on these parameters and their implications.
        world.Step(TIME_STEP, 10, 10)

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        # clock.tick(TARGET_FPS)
        inc += 1

    pygame.quit()

    print 'Total time: {0:.1f} s'.format(time.time()-time0)
    print 'Iterations: {0:d}'.format(len(velAverageRecord))
    # Plot average velocity history
    if autoStop and verbose:
        import matplotlib.pyplot as plt

        # plt.plot(timeRecord, velAverageRecord, 'bs-')
        plt.plot(timeRecord, velMaxRecord, 'r^-')
        plt.show()

    # print micro.fiber_volume()
    # micro.save_rve(filename='post.txt')
    # Update fibres position and plot final RVE
    for i, body in enumerate(fiberBodies):
        fiber = body.userData
        center = (body.position.x - margin_x, body.position.y - margin_y)
        if fiber.period in [1, 4]:
            # Find periodic fiber
            p = fiber.period
            fiberPeriodic = micro.findPeriodic(fiber)
            fiberPeriodic.set_center((center[0] + periodicSettings[p]['dx'],
                                      center[1] + periodicSettings[p]['dy']))
            fiberPeriodic.set_phi(fiberPeriodic.phi + body.angle * 180. / np.pi)
            fiberPeriodic.set_size(fiberPeriodic.L)  # Do not substract the tolerance of periodic fibres!

        fiber.set_center(center)
        fiber.set_phi(fiber.phi + body.angle * 180. / np.pi)
        fiber.set_size(fiber.L - tolerance)
    # micro.save_rve(filename='post_tol.txt')

    #micro.save_rve(filename=fileName+'-box2d_after')
    # micro.plot_rve(show_plot=True)

    return 0

def compactBox2D(micro, shake=False, margin=0.0, lmax=600.0, gravity=(0,-10), autoStop=True, crack=None, tolerance=0.1):

    if gravity == (0,0):
        velLimit = 5.0
    else:
        velLimit = 0.2

    maxDimension = float(max(micro.rveSize))
    PPM = lmax / maxDimension   # pixels per micron
    # print 'PPM = ', PPM

    RVE_WIDTH_PX = round(micro.rveSize[0] * PPM)
    RVE_HEIGHT_PX = round(micro.rveSize[1] * PPM)
    SCREEN_OFFSETX_PX, SCREEN_OFFSETY_PX = margin, margin

    SCREEN_WIDTH_PX = int(RVE_WIDTH_PX + 2*SCREEN_OFFSETX_PX)
    SCREEN_HEIGHT_PX = int(RVE_HEIGHT_PX + 2*SCREEN_OFFSETY_PX)

    # print 'Screen (pixels): %d x %d' % (SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
    # print 'RVE (pixels): %d x %d' % (RVE_WIDTH_PX, RVE_HEIGHT_PX)

    W = SCREEN_WIDTH_PX/PPM
    H = SCREEN_HEIGHT_PX/PPM
    margin_x, margin_y = SCREEN_OFFSETX_PX/PPM, SCREEN_OFFSETY_PX/PPM

    # --- pygame setup ---
    screen = pygame.display.set_mode((SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX), 0, 32)
    pygame.display.set_caption('Physical compaction')
    clock = pygame.time.Clock()

    # --- pybox2d world setup ---
    # Create the world
    world = b2World(gravity=gravity, doSleep=True)

    # Static edges to hold the fibers inside
    shapes =[
                b2EdgeShape(vertices=[      (margin_x, margin_y),   (margin_x, H-margin_y)]),
                b2EdgeShape(vertices=[    (margin_x, H-margin_y), (W-margin_x, H-margin_y)]),
                b2EdgeShape(vertices=[  (W-margin_x, H-margin_y),   (W-margin_x, margin_y)]),
                b2EdgeShape(vertices=[    (W-margin_x, margin_y),     (margin_x, margin_y)]),
            ]
    if crack:
        shapes.append(b2EdgeShape(vertices=[(margin_x, 0.5*H+margin_y), (crack+margin_x, 0.5*H+margin_y)]),)

    rve = world.CreateStaticBody(shapes=shapes)

    # Handler for periodic fibres
    periodicSettings = {
        # Parameters to generate the periodic fiber of a p1-fiber (p3)
        1: {'dx':   W, 'dy': 0.0, 'limit': H, #'localAnchor': (W,0),
            'moveAxis': (0,1), 'constraintAxis': (1,0)},

        # Parameters to generate the periodic fiber of a p4-fiber (p2)
        4: {'dx': 0.0, 'dy':   H, 'limit': W, #'localAnchor': (0,H),
            'moveAxis': (1,0), 'constraintAxis': (0,1)},
                        }

    # Not periodic fibers are dynamic bodies, otherwise they are static bodies
    # DONE - move periodic fibers
    # DONE - fiber as userdata for the bodies
    fiberBodies = list()
    numberOfDynamicBodies = 0
    # print 'Creating bodies'

    for fiber in micro.sq:
        center = (fiber.center[0]+margin_x, fiber.center[1]+margin_y)

        # Find closest fiber: fiberClosest. Too slow for high number of fibers
        # t1 = time.time()
        # try:
        #     fiberClosest = micro.findClosest(fiber)
        #     minDist = fiber.polygonly.distance(fiberClosest.polygonly)
        # except:
        #     minDist = 0.0
        # print 'Time for closest: %.2f' % (time.time()-t1)

        # minDist = np.random.normal(loc=0.5, scale=0.05)
        minDist = tolerance
        # print 'Time for distance: %.2f' % (time.time()-t1)

        if fiber.geometry.upper() == CIRCULAR:
            p = fiber.period

            if p == 0:
                body = world.CreateDynamicBody(position=center,
                                               # restitution=1.0,
                                               userData=fiber)
                numberOfDynamicBodies += 1
                body.CreateCircleFixture(radius=(fiber.L+minDist)*0.5,
                                         # density=1,
                                         friction=0.1)
                fiberBodies.append(body)
            elif p in [1,4]:
                body = world.CreateDynamicBody(position=center,
                                                # restitution=1.0,
                                                userData=fiber)
                body.CreateCircleFixture(radius=(fiber.L+minDist)*0.5,
                                         # density=1,
                                         friction=0.1)
                fiberBodies.append(body)

                # Find periodic fiber
                fiberPeriodic = micro.findPeriodic(fiber)
                centerP = (center[0]+periodicSettings[p]['dx'], center[1]+periodicSettings[p]['dy'])
                periodicBody = world.CreateDynamicBody(position=centerP,
                                                        # restitution=1.0,
                                                        userData=fiberPeriodic)
                periodicBody.CreateCircleFixture(radius=(fiber.L+minDist)*0.5,
                                                 # density=1.,
                                                 friction=0.1)
                fiberBodies.append(periodicBody)

                # Implement periodic constraint

                if p == 1:
                    c  = (center[0],0)
                    cp = (centerP[0],0)
                else: # p = 4
                    c  = (0, center[1])
                    cp = (0, centerP[1])

                # Fiber1 - RVE
                world.CreatePrismaticJoint(
                    bodyA=rve,
                    bodyB=body,
                    enableLimit=True,
                    localAnchorA=c,
                    localAnchorB=(0, 0),
                    axis=periodicSettings[p]['moveAxis'],
                    lowerTranslation=0,
                    upperTranslation=periodicSettings[p]['limit'],
                    )

                # Fiber2 - RVE
                world.CreatePrismaticJoint(
                    bodyA=rve,
                    bodyB=periodicBody,
                    enableLimit=True,
                    localAnchorA=cp,
                    localAnchorB=(0, 0),
                    axis=periodicSettings[p]['moveAxis'],
                    lowerTranslation=0,
                    upperTranslation=periodicSettings[p]['limit'],
                    )

                # Fiber1- Fiber2. Prismatic joint
                world.CreatePrismaticJoint(
                    bodyA=body,
                    bodyB=periodicBody,
                    # enableLimit=True,
                    localAnchorA=(0, 0),
                    localAnchorB=(0, 0),
                    axis=periodicSettings[p]['constraintAxis'],
                    # lowerTranslation=W,
                    # upperTranslation=W,
                    )

                # Fiber1- Fiber2. Distance joint
                world.CreateDistanceJoint(
                    bodyA=body,
                    bodyB=periodicBody,
                    anchorA=(0, 0),
                    anchorB=(0, 0),
                    # frequencyHz = 30.0,
                    # dampingRatio = 0.5,
                    collideConnected=True)

            elif p in [5,6,7,8]:
                    body = world.CreateStaticBody(position=center,
                                                   # restitution=1.0,
                                                   userData=fiber)
                    body.CreateCircleFixture(radius=(fiber.L+minDist)*0.5,
                                             # density=1.,
                                             friction=0.1)
                    fiberBodies.append(body)
        else:
            raise TypeError, 'Only CIRCULAR fibers'
        #     vertices = zip(*fiber.vertices)
        #     myshape = []
        #     for i in range(0,len(vertices)-1):
        #         s = b2PolygonShape(vertices=[vertices[i], vertices[i+1], center])
        #         myshape.append(s)
        #
        #     if (fiber.period == 0):# and (np.random.random()>0.2):
        #         body = world.CreateDynamicBody(#position=c, angle=15,
        #                                      shapeFixture=b2FixtureDef(density=1, friction=0.1),
        #                                      shapes=myshape, restitution=1.0, userData=fiber)
        #         numberOfDynamicBodies += 1
        #     else:
        #         body = world.CreateStaticBody(#position=c, angle=15,
        #                                      shapeFixture=b2FixtureDef(density=1, friction=0.1),
        #                                      shapes=myshape, restitution=1.0, userData=fiber)
        # fiberBodies.append(body)

    # --- main game loop ---
    running = True
    time0 = time.time()
    time2stop = 0.1
    velAverageRecord = list()
    velMaxRecord = list()
    timeRecord = list()

    # print 'Entering main loop'

    while running:
        # Check the event queue
        for event in pygame.event.get():
            if event.type==QUIT or (event.type==KEYDOWN and event.key==K_ESCAPE):
                # The user closed the window or pressed escape
                running = False

        screen.fill(colors['background'])

        # Draw the world
        for body in fiberBodies:
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                # velocity += body.linearVelocity.length
                try:
                    fixture.shape.draw(body, PPM, SCREEN_HEIGHT_PX, screen)
                except:
                    print 'Could not draw body'

        if shake:
            if 10.0>time.time()-time0>0.5*time2stop:
                world.gravity = (np.exp(-(time.time()-time0))*10000.0*np.sin(100.0*time.time())+gravity[0], gravity[1])
            else:
                world.gravity = gravity

        if (time.time()-time0 >= time2stop) and autoStop:
            velAverage = sum([body.linearVelocity.length for body in fiberBodies])/numberOfDynamicBodies
            velMax = max([body.linearVelocity.length for body in fiberBodies])
            # print 'Average velocity %f' % velocity
            velAverageRecord.append(velAverage)
            velMaxRecord.append(velMax)
            timeRecord.append(time2stop)
            if (velAverage < velLimit) and (velMax < 1.0): # or (velocity < max(velocityRecord)/50.0):
                running = False
            time2stop += 0.1

        for body in (rve,):
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                fixture.shape.draw(body, PPM, SCREEN_HEIGHT_PX, screen)

        # Make Box2D simulate the physics of our world for one step.
        # Instruct the world to perform a single step of simulation. It is
        # generally best to keep the time step and iterations fixed.
        # See the manual (Section "Simulating the World") for further discussion
        # on these parameters and their implications.
        world.Step(TIME_STEP, 10, 10)

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()


    # # Plot average velocity history
    # if autoStop:
    #     import matplotlib.pyplot as plt
    #     plt.plot(timeRecord, velAverageRecord, 'bs-')
    #     plt.plot(timeRecord, velMaxRecord, 'r1-')
    #     plt.show()

    # Update fibres position and plot final RVE
    for i, body in enumerate(fiberBodies):
        fiber = body.userData
        if micro.sq[i].geometry.upper() == CIRCULAR:
            center = (body.position.x-margin_x, body.position.y-margin_y)
            fiber.set_center(center)
            # micro.sq[i].set_size(micro.sq[i].L)

    # Check fibers position. Enforce periodic fibers. Periodic fibers must be accurately periodic
    # Solve possible overlapping of fibers
    # t0 = time.time()
    micro.validate() # TODO - check this function. It is not robust when checking periodic fibers
    micro.setPeriodicity()
    # print 'Time to correct microstructure: %.2f s' % (time.time()-t0)

    # micro.plot_rve()
    print('Done!')

########################################################################################
## Microstructure auxiliary functions

def periodic_layout(S0, Vf, fiber, pattern='SQUARE', tolerance=0.1, optPeriod=FULL):
    """
    Optimization algorithm to produce periodical microstructures
    :param S0: RVE dimensions
    :param Vf: fiber volume fraction (/1)
    :param fiber: model fiber
    :param L: fiber size
    :param layout: type of periodic arrangement: 'SQUARE' or 'HEXAGONAL'
    :return: horizontal (l1) and vertical (l2) spacing
    """
    if SQUARE in pattern.upper():
        pattern = SQUARE
    elif HEXAGONAL in pattern.upper():
        pattern = HEXAGONAL
    else:
        raise ValueError, 'Specify layout arrangement'

    if Vf > 1.0: Vf /= 100.0

    L, H = S0

    # Use the attribute 'bounds' from class Polygon (shapely)
    minx, miny, maxx, maxy = fiber.polygonly.bounds
    lx = maxx - minx + tolerance
    ly = maxy - miny + tolerance
    # TODO lx and ly may be improved through shapely for rotated fibers

    # Hexagonal regular layout does not support not horizontal periodicity
    if optPeriod in [HORIZ, NONE] and (pattern == 'HEXAGONAL'):
        print '%s layout cannot be %s or %s' % (HEXAGONAL, HORIZ, NONE)

    # Offset from edges if optPeriod is not FULL
    if optPeriod == FULL:
        dx = 0.0
        dy = 0.0
    elif optPeriod == HORIZ:
        dx = lx*0.5
        dy = 0.0
    elif optPeriod == VERT:
        dx = 0.0
        dy = ly*0.5
    elif optPeriod == NONE:
        dx = lx*0.5
        dy = ly*0.5
    else:
        raise ValueError, 'Incorrect periodicity definition: %s, %s, %s or %s' % (FULL, HORIZ, VERT, NONE)
    print 'dx = %.3f, dy = %.3f' % (dx, dy)

    A = fiber.polygonly.area  #np.pi/4*df**2
    print 'Fiber area = %.2f%%/fiber' % (A/(L*H)*100)
    N = Vf * L * H / A
    print 'Number of fibers required: %.1f, Vf = %.1f %%' % (N, Vf * 100)

    L_eff = L - 2*dx
    H_eff = H - 2*dy

    if pattern == SQUARE:
        nx = round(L_eff * np.sqrt(Vf / A * ly / lx))
        ny = round(H_eff * np.sqrt(Vf / A * lx / ly))

        # Check horizontal spacing
        if nx >= L_eff / lx:
            nx = int(L_eff / lx)
        l1 = L_eff / nx  # l1 > dx

        # Check vertical spacing
        if ny >= H_eff / ly:
            ny = int(H_eff / ly)
        l2 = H_eff / ny  # l2 > dy

        # Real volume fraction
        nx_ = nx
        if dx: nx_ = nx + 1
        ny_ = ny
        if dy: ny_ = ny + 1
        Nt = nx_ * ny_

    elif pattern == HEXAGONAL:
        nx = round(L_eff * np.sqrt(ly / lx * (Vf * np.sin(np.pi / 3.) / A)))
        ny = round(H_eff * np.sqrt(lx / ly * Vf / (A * np.sin(np.pi / 3.))))

        # Check horizontal spacing
        if nx >= L_eff / lx:
            nx = int(L_eff / lx)
        l1 = L_eff / nx  # l1 > dx

        # Check vertical spacing
        if ny >= H_eff / ly:
            ny = int(H_eff / (ly * np.sin(np.pi / 3.)))
        if not dy and (ny % 2 != 0):
            ny -= 1  # Last row equals first row
        elif dy and (ny % 2 == 0):
            ny -= 1  # last row different of first row
        l2 = H_eff / ny  # l2 > dy
        # Real volume fraction
        nx_ = nx
        if dx: nx_ = nx + 1
        ny_ = ny
        if dy: ny_ = ny + 1
        if dx:
            Nt = (2*nx_-1)*ny_*0.5
        else:
            Nt = nx_ * ny_

    else:
        raise ValueError, 'Incorrect pattern definition: %s or %s' % (SQUARE, HEXAGONAL)

    # print nx, ny    # Number of spaces
    print 'n_x = %d, n_y = %d' % (nx_, ny_)  # Number of fibers

    periodicVf = Nt * A / (L * H)
    print 'Periodic fibers: %d, Vf = %.1f %%' % (Nt, periodicVf * 100)

    # list of centers
    centers = []
    for i in range(int(ny + 1)):
        centers.append([])
        y = dy + i * l2
        if (i % 2 == 1) and (pattern == HEXAGONAL):
            # even rows include one fiber less in hexagonal pattern
            for j in range(int(nx)):
                x = dx + l1 * (0.5 + j)
                centers[-1].append((x, y))
            # if dx:  # Artificial fibers
            #     centers[-1].append((0, y))
            #     centers[-1].append((L, y))
        else:
            for j in range(int(nx + 1)):
                x = dx + j * l1
                centers[-1].append((x, y))
                # print centers[-1]

    return centers

def copy_RVE(sq, notPeriodic=False):
    """
    Function: COPY_RVE
    """
    if notPeriodic:
        return [Fiber(geometry=f.geometry, parameters=f.parameters, L=f.L, phi=f.phi, center=f.center,
                      period=f.period, Nf=f.Nf, material=f.material) for f in sq if f.Nf==1]
    else:
        return [Fiber(geometry=f.geometry, parameters=f.parameters, L=f.L, phi=f.phi, center=f.center,
                      period=f.period, Nf=f.Nf, material=f.material) for f in sq]

def patternedMicrostructure(microstructure, ncols=2, nrows=1, fileout='', directory='.', save=True, plot=True):
    """
    Generates a patterned microstructure of a given microstructure.
    :param microstructure: txt with the microstructure information or Microstructure object
    :param ncols: default is 2
    :param nrows: default is 1
    :return Patterned Microstructure:
    """

    if isinstance(microstructure, str):
        directory, filename = os.path.split(microstructure)
        # Read initial microstructure
        myMicro = Microstructure(read_microstructure=[directory, filename])
    elif isinstance(microstructure, Microstructure):
        myMicro = microstructure
    else:
        raise TypeError, 'Expected path to text file or Microstructure object: got %s' % type(microstructure)

    dx = myMicro.rveSize[0]
    dy = myMicro.rveSize[1]
    delta_x, delta_y = 0.0, 0.0

    # Generate empty microstructure with the target width (ncols)
    PatternedMicrostructure = Microstructure(rve_size=(dx * ncols, dy))
    PatternedMicrostructure.sq = copy_RVE(myMicro.sq)

    # I. Produce horizontal pattern (ncols)
    for i in range(1, ncols):

        delta_x += dx

        for f in myMicro.sq:
            newCenter = (f.center[0] + delta_x, f.center[1])

            if f.period in [0, 2, 3, 4, 7, 8]:
                newFiber = Fiber(geometry=f.geometry, parameters=f.parameters, L=f.L, phi=f.phi, center=newCenter,
                                 period=0, Nf=1, material=f.material)
                PatternedMicrostructure.sq.append(newFiber)


    # Manage fibers period and Nf
    PatternedMicrostructure.setPeriodicity()
    tempMicro = copy_RVE(PatternedMicrostructure.sq)  # Temporary list of fibres
    # Resize PatternedMicrostructure
    PatternedMicrostructure.rveSize = (dx * ncols, dy * nrows)

    # II. Produce vertical pattern (nrows)
    for j in range(1, nrows):

        delta_y += dy

        for f in tempMicro:
            newCenter = (f.center[0], f.center[1] + delta_y)

            if f.period in [0, 1, 2, 3, 6, 7]:
                newFiber = Fiber(geometry=f.geometry, parameters=f.parameters, L=f.L, phi=f.phi, center=newCenter,
                                 period=0, Nf=1, material=f.material)
                PatternedMicrostructure.sq.append(newFiber)


    # Manage fibers period and Nf
    PatternedMicrostructure.setPeriodicity()

    if save:
        if not fileout and isinstance(microstructure, str):
            fileout = 'Patterned-' + fileout
        PatternedMicrostructure.save_rve(filename=fileout, directory=directory)

    if plot:
        PatternedMicrostructure.plot_rve(numbering=False, mute=False, matrix_colour='firebrick', fibre_colour='')

    return PatternedMicrostructure

def moment_of_inertia(vertices, center):
    "Last vertex must coincide with first vertex"
    x = vertices[0] - center[0]
    y = vertices[1] - center[1]
    N = len(x)
    Ix, Iy = 0.0, 0.0
    for i in range(N - 1):
        Ix += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) * (y[i + 1] * y[i + 1] + y[i] * y[i])
        Iy += (y[i + 1] - y[i]) * (x[i + 1] + x[i]) * (x[i + 1] * x[i + 1] + x[i] * x[i])
    Ix = abs(Ix / 12.0)
    Iy = abs(Iy / 12.0)
    Iz = Ix + Iy
    return Ix, Iy, Iz

def plot_distribution(data, title='Statistical distribution'):
    # s = np.random.normal(size=N)   # generate your data sample with N elements

    if isinstance(data, list): data = np.array(data)

    mu = data.mean()
    sigma = data.std()

    fig, ax = plt.subplots()
    n = 25

    count, bins, ignored = ax.hist(data, n, normed=True)
    ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
             linewidth=2, color='r')
    # ax.hist(data, bins=n) # bin it into n = N/10 bins
    # ax.set_xlim(0, max(1.0, 1.05*max(data)))
    ax.set_title(title)
    plt.show()

def colourFiber(material='CF'):
    if 'CF' in material.upper():
        fibercolour = 'silver'
        textcolour = 'black'
        edgecolour = textcolour
    elif 'GF' in material.upper():
        fibercolour = 'beige'
        textcolour = 'black'
        edgecolour = textcolour
    elif 'AF' in material.upper():
        fibercolour = 'khaki'
        textcolour = 'black'
        edgecolour = textcolour
    elif 'PF' in material.upper():
        fibercolour = 'cadetblue'
        textcolour = 'white'
        edgecolour = textcolour
    elif 'VOID' in material.upper():
        fibercolour = 'white'
        textcolour = 'grey'
        edgecolour = 'white'
    else:
        print material + " is not in the default colours list\n"
        fibercolour = 'palegreen'
        textcolour = 'black'
        edgecolour = textcolour
    return fibercolour, textcolour, edgecolour

def donut(center, r, dr, resolution=1000):
    """
    Create donut-like polygon
    :param center:
    :param r:
    :param dr:
    :return:
    """
    c = Point(center).buffer(r+dr, resolution=resolution)
    c_ = Point(center).buffer(r, resolution=resolution)
    return c.difference(c_)

def circularIntersection(center, r, fiber, resolution=1000):
    c = Point(center).buffer(r, resolution=resolution)
    return c.intersection(fiber.polygonly)

def distLine2Point(Q, v, P):
    """
    Q and v define the line
    P is the point
    :return: the distance between the point and the line
    """

    r = P[0]-Q[0], P[1]-Q[1]  # vector PQ

    v_ = np.linalg.norm(v)
    n = v[1]/v_, -v[0]/v_  # normal to 'v' with module 1

    # Dot product
    return abs(np.dot(n,r))

def getQuadrant(v):
    x, y = v
    if x>=0:
        if y>= 0:
            return 1
        else:
            return  4
    else:
        if y>=0:
            return 2
        else:
            return 3

def generateRandomNormal(L, dL, notNegative=True):

    if dL<=0:
        return L

    l = np.random.normal(loc=L, scale=dL)

    if l<=0 and notNegative:
        return L*0.5
    else:
        return l

def dist2points(A, B):
    """ Distance between two points (2D) """
    return np.sqrt((A[0]-B[0])*(A[0]-B[0]) + (A[1]-B[1])*(A[1]-B[1]))

def lvf(micro, x, y, radius):
    """
    Computes local volume fraction on a point (x, y)
    :param micro: microstructure
    :param x: x-coordinate of the reference point
    :param y: y-coordinate of the reference point
    :param radius: spot size
    :return: local volume fraction
    """
    A = np.pi*radius*radius
    circle = Point(x, y).buffer(radius)
    Af = 0.0

    fibres = [f for f in micro.sq if dist2points((x,y),(f.center)) < radius+f.L*0.5]

    for fibre in fibres:
        # Check intersection
        if dist2points((x,y),(fibre.center))+fibre.L*0.5 < radius:
            # Full intersection
            Af += fibre.polygonly.area
        else:
            # Compute intersection
            Af += circle.intersection(fibre.polygonly).area

    # Af = sum([circle.intersection(fibre.polygonly).area for fibre in micro.sq if (dist2points((x,y),(fibre.center))-fibre.L*0.5) < radius])

    return Af/A

def poly2(x, A1, B1, C1):
    return A1*x*x + B1*x + C1

def formatTicks(axes, fontsize=16, x=True, y=True, z=False, color='black'):

    if x:
        for t1 in axes.get_xticklabels():
            t1.set_color(color)
            t1.set_size(fontsize)
    if y:
        for t1 in axes.get_yticklabels():
            t1.set_color(color)
            t1.set_size(fontsize)

    if z:
        for t1 in axes.get_yticklabels():
            t1.set_color(color)
            t1.set_size(fontsize)

    return 0

# Optimize G-function calculus
def annular_intersection(center, r, dr, fibers, distances):

    partialFibers = [f for f in fibers if distances[f][0] < r < distances[f][1]]

    g = sum([donut(center, r, dr).intersection(f.polygonly).area / (f.polygonly.area) for f in partialFibers])

    return g

## Paper: Potential generation
def statisticGeneration(L=50., H=50., num=10, filename=r'POT_Vf65', algorithm=POTENTIAL,
                        directory=r'C:\Users\miguel.herraez\Desktop\VIPPER project\TO DO potential dispersion\Analysis',
                        fibersets=[{'Geometry': u'Circular', 'Phi': (0.0, 0.0), 'Parameters': [], 'Vf': 50., 'df': (5., 0.0), 'Material': u'CF-AS4'}],):

    os.chdir(directory)

    # Change standard output
    import sys
    saveout = sys.stdout
    fsock = open(filename+'.log', 'w')
    sys.stdout = fsock

    ##
    rve_size = (L, H)
    rmax = 0.95*max(rve_size)
    dr = 0.1
    all_g = []
    all_K = []
    micros = []
    i = 0
    t0 = time.time()
    while i < num:
        # Generate microstructure
        while True:
            # fibersets = [# {'Geometry': u'CSHAPE', 'Phi': (0.0, 180.0), 'Parameters': [0.2, 30.0], 'Vf': 44.0, 'df': (6.19, 0.25), 'Material': u'CF-AS4'},
            #      {'Geometry': u'Circular', 'Phi': (0.0, 0.0), 'Parameters': [], 'Vf': Vf*100., 'df': (d, 0.0), 'Material': u'CF-AS4'},
            #      ]
            # try:
            myMicro = Microstructure(rve_size=rve_size, fibersets=fibersets, gen_algorithm=algorithm, tolerance=0.1)
            # except:
            #     print '-**- Error during generation -**-'
            #     continue
            currentVf = myMicro.fiber_volume()[0]
            myMicro.plot_rve(filename='{0:s}_{1:03d}'.format(filename, i+1), directory=directory, numbering=False, save=True,
                              matrix_colour='', fibre_colour='', show_plot=False)
            if 100.*currentVf >= 0.99*fibersets[0]['Vf']:
                break
            else:
                print 'Vf not reached: {0:.1f}% < {1:.1f}%'.format(currentVf*100., fibersets[0]['Vf'])

        # Save microstructure
        myMicro.save_rve(filename='{0:s}_{1:03d}'.format(filename, i+1))
        i += 1

    t1 = time.time()
    print 'Generation time: {0:.1f} s. (Average = {1:.1f} s.)'.format(t1-t0, (t1-t0)/num)

    for i in range(num):
        myMicro = Microstructure(read_microstructure='{0:s}_{1:03d}'.format(filename, i+1))
        try:
            # Distribution analysis
            print '      Analysis {}'.format(i+1)
            r, g, K = myMicro.radialDistFunction(rmax=rmax, dr=dr, plot=False, verbose=False)
        except:
            print 'Analysis failed'
            continue
        all_g.append(g)
        all_K.append(K)
        micros.append(myMicro)
        i += 1
    t2 = time.time()
    print 'Analysis time: {0:.1f} s. (Average = {1:.1f} s.)'.format(t2-t1, (t2-t1)/num)

    # Statistic analysis of the distributions
    all_g = np.array(all_g)
    all_K = np.array(all_K)
    mean_g = all_g.mean(axis=0)
    error_g = all_g.std(axis=0)
    mean_K = all_K.mean(axis=0)
    error_K = all_K.std(axis=0)

    ## WRITE. Statistical results
    f = open(filename+'_stats.txt','w')
    f.write('{0:>6s} {1:>9s} {2:>6s} {3:>6s} {4:>6s}'.format('r', 'K(r)', 'dK(r)', 'G(r)', 'dG(r)'))
    for r_,k_,g_,dk_,dg_ in zip(r, mean_K, mean_g, error_K, error_g):
        f.write('\n{0:6.3f} {1:9.2f} {2:6.3f} {3:6.2f} {4:6.3f}'.format(r_,k_,dk_,g_,dg_))
    f.close()

    ## PLOT
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 8), dpi=100, facecolor='w', edgecolor='k')
    ax[0].plot(r, mean_K, 'b-', lw=1.)
    ax[0].errorbar(r[::4], mean_K[::4], yerr=error_K[::4], fmt='x', lw=1., ms=3., ecolor='k')
    ax[0].set_xlim(0, r[-2])
    ax[0].set_ylim(0, 1.1*max(mean_K))
    # ax[0].set_xlabel(r'$r\ (\mu m)$', fontsize=16)
    ax[0].set_ylabel(r'$K(r)$', fontsize=20)
    ax[0].set_title(r'Second-order intensity function')
    # leg0 = ax[0].legend()
    # leg0.draggable()

    ax[1].plot(r[:-1], mean_g[:-1], 'b-', lw=1.)
    ax[1].errorbar(r[:-1:4], mean_g[:-1:4], yerr=error_g[:-1:4], fmt='x', ms=3., ecolor='k', mec='k')
    ax[1].set_xlim(0, r[-2])
    ax[1].set_ylim([0, 1.1*max(mean_g)])
    ax[1].set_xlabel(r'$r\ (\mu m)$', fontsize=20)
    ax[1].set_ylabel(r'$G(r)$', fontsize=20)
    ax[1].set_title(r'Radial distribution function')
    ax[1].axhline(y=1.0, xmin=0, xmax=r[-1], color='r', ls='--', lw=1.)
    # leg1 = ax[1].legend()
    # leg1.draggable()

    # Restore standard output
    sys.stdout = saveout
    fsock.close()

    # plt.show()



    ##


#--End of functions----------------------------------------------------------------


def main():
    directory = r'C:\Users\miguel.herraez\Desktop\VIPPER project\Microstructures'
    # directory = r'C:\Users\miguel.herraez\Desktop\VIPPER project\_DONE potential dispersion\Analysis'
    # plt.ion()

    if 1:  # Test generation
        os.chdir(directory)
        rve_size = (100.0, 60.0)

        # Define fibersets
        fibersets = [
            # {'Geometry': u'CSHAPE',    'Phi': 180.0, 'Parameters': [0.6, 270.0], 'Vf': 40.0, 'df': 3.19, 'Material': u'PF-POLYAMIDE'},
            # {'Geometry': u'CSHAPE',    'Phi': 180.0, 'Parameters': [0.4, 135.0], 'Vf': 10.0, 'df': 5.19, 'Material': u'GF-S2'},
            # {'Geometry': u'Lobular',    'Phi': (0.,90.), 'Parameters': [2     ], 'Vf': 60.0, 'df': (7.19,0), 'Material': u'PF-POLYAMIDE'},
            # {'Geometry': u'CSHAPE', 'Phi': (0.0, 180.0), 'Parameters': [0.2, 30.0], 'Vf': 44.0, 'df': (6.19, 0.25), 'Material': u'CF-AS4'},
            {'Geometry': u'Circular',   'Phi': (0.0,0.0),  'Parameters': [],       'Vf': 50.0, 'df': (7.2,0), 'Material': u'CF-AS4'},
            # {'Geometry': u'Circular',   'Phi': (0.0,0.0),  'Parameters': [],       'Vf': 20.0, 'df': (5.,0), 'Material': u'GF-S'},
            # {'Geometry': u'Elliptical',   'Phi': 15.0, 'Parameters': [0.85],   'Vf': 70.0, 'df': 7.19, 'Material': u'AF-AS4'},
            # {'Geometry': u'Oval',         'Phi': 15.0, 'Parameters': [3, 0.3], 'Vf': 90.0, 'df': 7.19, 'Material': u'GF-S2'},
            # {'Geometry': u'Spolygonal',   'Phi': 15.0, 'Parameters': [4, 0.1], 'Vf': 90.0, 'df': 7.19, 'Material': u'CF-AS4'},
        ]
        tinit = time.time()
        # myMicrostructure = Microstructure(rve_size=rve_size, fibersets=fibersets, gen_algorithm=NNA,
        #                                   comp_algorithm=[POINT, (0, 0)], tolerance=0.1)

        filename = r'test_spiral'
        myMicrostructure = Microstructure(rve_size=rve_size, fibersets=fibersets, gen_algorithm=DYNAMIC, tolerance=0.35,
                                          optPeriod=HORIZ)
        print 'Time: %5.1f s.' % (time.time() - tinit)
        # myMicrostructure.save_rve(filename=filename, directory=os.path.join(directory,'Yang_Cshape'))
        myMicrostructure.plot_rve(filename=filename, directory=directory, numbering=True, save=False,
                                  matrix_colour='', fibre_colour='', show_plot=True)
        compactBox2D(myMicrostructure, shake=False)
        # patternedMicrostructure(myMicrostructure, ncols=2, nrows=1, fileout='Patterned-' + filename, directory=directory)

    elif 0: # Test compaction
        filename = r'ttt'
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename), tolerance=0.25)
        # myMicrostructure.plot_rve(filename=filename, directory=directory, numbering=True, save=True, mute=False)
        tinit = time.time()
        # myMicrostructure.compact_RVE(algorithm=DIRECTION, vector=(0.0, -1.0))
        myMicrostructure.stirringAlgorithm(eff=1.)
        print 'Time: %5.1f s.' % (time.time() - tinit)
        # Plot and save resulting microstructure
        myMicrostructure.plot_rve(filename=filename + 'comp', directory=directory, numbering=True, save=True, mute=False)
        # myMicrostructure.save_rve(filename=filename + 'comp', directory=directory, analysis=False)

    elif 0: # Test rectangular layout
        filename = r'test-pattern'
        rve_size = (30.0, 30.0)
        # Define fibersets
        fibersets = [
            {'Geometry': u'Circular', 'Phi': 0.0, 'Parameters': [], 'Vf': 60.0, 'df': 7.19, 'Material': u'GF-S2'},
            # {'Geometry': u'Circular', 'Phi': 0.0, 'Parameters': [], 'Vf': 10.0, 'df': 4.19, 'Material': u'CF-AS4'},
            # {'Geometry': u'Elliptical', 'Phi': 15.0, 'Parameters': [0.85], 'Vf': 10.0, 'df': 7.19,
            #  'Material': u'AF-AS4'},
            # # {'Geometry': u'Oval',         'Phi': 15.0, 'Parameters': [3, 0.3], 'Vf': 90.0, 'df': 7.19, 'Material': u'GF-S2'},
            # {'Geometry': u'Spolygonal', 'Phi': 15.0, 'Parameters': [4, 0.1], 'Vf': 10.0, 'df': 5.19,
            #  'Material': u'CF-AS4'},
        ]
        tinit = time.time()
        # myMicrostructure = Microstructure(rve_size=rve_size, fibersets=fibersets, gen_algorithm=NNA,
        #                                   comp_algorithm=(ABSORPTION, (0, 0)), tolerance=0.1)
        myMicrostructure = Microstructure(rve_size=rve_size, fibersets=fibersets, gen_algorithm='PERIODIC-HEXAGONAL',
                                          tolerance=0.1, optPeriod=NONE)
        print 'Time: %5.1f s.' % (time.time() - tinit)
        myMicrostructure.save_rve(filename=filename, directory=directory)
        myMicrostructure.plot_rve(filename=filename, directory=directory, numbering=False, save=True, matrix_colour='firebrick',
                                  fibre_colour='', mute=False)

        # patternedMicrostructure(myMicrostructure, ncols=2, nrows=3, fileout='Patterned-' + filename,
        #                         directory=directory)

    elif 0: # Test analysis: diameters, nearest neighbour, center of mass, inertia
        filename = r'NewMicrostructure_2comp'
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename))

        # Clear microstructure from errors
        # myMicrostructure.clearMicrostructure()
        # myMicrostructure.save_rve(filename=filename, directory=directory)

        # Plot and save resulting microstructure
        myMicrostructure.plot_rve(filename=filename, directory=directory, numbering=False, save=True,
                                  COM=True, MOI=True, show_plot=True, mute=False)
        # tinit = time.time()
        # print myMicrostructure.analyzeCenterOfMass()
        # print myMicrostructure.analyzeMomentsOfInertia()
        # ds = myMicrostructure.analyzeNearestNeighbour(show_plot=True, neighbour=1)
        dL = myMicrostructure.analyzeDiameters(show_plot=True)
        print dL
        # print 'Time: %5.1f s.' % (time.time()-tinit)
        # print ds

    elif 0: # Test validation of RVE
        filename = r'kink_periodic_L60x120_Vf58'
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename))

        mc = "#60a4f3"
        fc = "#f89447"
        myMicrostructure.plot_rve(numbering=True, matrix_colour=mc, fibre_colour=fc, mute=True, title=False)

        myMicrostructure.validate()
        myMicrostructure.setPeriodicity()

        myMicrostructure.plot_rve(numbering=True)

        myMicrostructure.save_rve(filename='kink_periodic_L60x120_Vf58')

    elif 0: # Set microstructure for SSY crack model
        filename = r'pyMS_circ_N43_Vf50_A'
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename))
        # myMicrostructure.validate()
        # myMicrostructure.save_rve(filename=r'lob2_Vf63_SSY07', directory=directory, analysis=False)
        myMicrostructure.plot_rve(title=False, numbering=False, matrix_colour='firebrick', fibre_colour='lightyellow')

    elif 0: # Test local volume fraction function
        filename = r'01'
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename))
        x,y,myLvf = myMicrostructure.localVolumeFraction(spotSize=30., voxelSize=1., spots=True, fibres=True)
        # myMicrostructure.plot_rve(mute=False, show_plot=False, save=True, filename='probando', directory=directory)
        print 'Bounds: {0:.1f}% < Vf < {1:.1f}%'.format(np.min(myLvf)*100, np.max(myLvf)*100)
        # plt.ion()
        plt.show()

    elif 0: # Voronoi statistics
        # filename = r'testBox2d_squares_circles'
        # filename = r'POT_300x300_Vf65_004'
        filename = r'01'
        # plt.ion()
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename))
        mean, std = myMicrostructure.voronoi()

        print 'Voronoi analysis: Mean = {0:.2f}, Std = {1:.2f}'.format(mean, std)
        ##

    elif 0: # Test Second-order intensity function and radial distribution function (Segurado)
        filename = r'lob2'
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename))
        rmax = 50.
        dr = .0
        t = time.time()
        r, g1, K1 = myMicrostructure.radialDistFunction(rmax=rmax, dr=dr, plot=False, strategy=1)
        t1 = time.time()-t
        print t1

        t = time.time()
        _, g3, K3 = myMicrostructure.radialDistFunction(rmax=rmax, dr=dr, plot=False, strategy=3)
        t3 = time.time()-t
        print t3

        fig, ax = plt.subplots(ncols=2)
        ax[0].plot(r, K1, 'b-', lw=2, label='K discrete. {0:.1f} s'.format(t1))
        ax[0].plot(r, K3, 'r-', lw=2, label='K continuous. {0:.1f} s'.format(t3))
        ax[0].plot(r, np.pi*r*r, 'k--', lw=1)
        ax[1].plot(r, g1, 'b-', lw=2, label='g discrete')
        ax[1].plot(r, g3, 'r-', lw=2, label='g continuous')
        ax[1].axhline(y=1.0, xmin=0, xmax=r[-1], color='k', ls='--', lw=1., label='Reference')
        leg = ax[0].legend()
        leg.draggable()

        plt.show()
        #

    elif 0: # Test Second-order intensity function (K)
        filename = r'11'
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename))
        L, H = myMicrostructure.rveSize
        rmax = 20.
        dr = 0.1
        r = np.arange(0,rmax,dr)
        Na = sum([f.Nf for f in myMicrostructure.sq])/(L*H)

        # Check K
        t = time.time()
        _, K = myMicrostructure.secondOrder_K(dr=dr, rmax=rmax)
        print 'K. Time: {} s'.format(time.time()-t)
        ##

        pmicro = patternedMicrostructure(myMicrostructure, ncols=3, nrows=3, plot=False, save=False)

        referenceFiber = myMicrostructure.sq[36]
        cx, cy = referenceFiber.center
        referenceFiber_pattern = sorted([(f, dist2points(f.center, (cx+L, cy+H))) for f in pmicro.sq], key=lambda x: x[1])[0][0]

        #
        t = time.time()
        I1 = myMicrostructure._intensity(fiber=referenceFiber_pattern, r=r, strategy=1, pmicro=pmicro)
        t1 = time.time()-t
        print 'Strategy 1. Time: {} s'.format(t1)
        #
        t = time.time()
        I2 = myMicrostructure._intensity(fiber=referenceFiber_pattern, r=r, strategy=2, pmicro=pmicro)
        t2 = time.time()-t
        print 'Strategy 2. Time: {} s'.format(t2)
        #
        t = time.time()
        I3 = myMicrostructure._intensity(fiber=referenceFiber_pattern, r=r, strategy=3, pmicro=pmicro)
        t3 = time.time()-t
        print 'Strategy 3. Time: {} s'.format(t3)
        #
        t = time.time()
        I4 = myMicrostructure._intensity(fiber=referenceFiber_pattern, r=r, strategy=4, pmicro=pmicro)
        t4 = time.time()-t
        print 'Strategy 4. Time: {} s'.format(t4)
        #
        t1e = time.time()
        I1e = myMicrostructure._Eintensity(fiber=referenceFiber, r=r, strategy=1)
        print 'Strategy 1 (edge effect). Time: {} s'.format(time.time()-t1e)
        # Plot
        plt.plot(r,     K, 'y-', lw=2, label='K discrete')
        plt.plot(r, I1/Na, 'r:', lw=2, label='Discrete: {0:.3f} s.'.format(t1))
        plt.plot(r, I2/Na, 'b--', lw=2, label='Continuous: {0:.3f} s.'.format(t2))
        plt.plot(r, I3/Na, 'k', marker='s', mfc='w', label='Continuous enhanced: {0:.3f} s.'.format(t3))
        plt.plot(r, I4/Na, 'm', marker='^', mfc='m', label='Continuous parallel: {0:.3f} s.'.format(t4))
        plt.plot(r, I1e/Na, 'c-.', lw=2, label='Discrete (edge effect)')

        leg = plt.legend(numpoints=1)
        leg.draggable()

        # r1, K1 = myMicrostructure.secondOrder_K(dr=dr, rmax=rmax, verbose=False, strategy=1)
        # r2, K2 = myMicrostructure.secondOrder_K(dr=dr, rmax=rmax, verbose=False, strategy=2)
        # plt.plot(r1, K1, 'r:')
        # plt.plot(r2, K2, 'b--')
        # plt.plot(r, np.pi*r**2, 'g')
        #
        # # Fit to polynomial (quadratic)
        # from scipy.optimize import curve_fit
        # popt, _ = curve_fit(poly2, r, K, p0=(1.0, 1.0, -1.0))
        # Afit, Bfit, Cfit = popt
        # print 'Fitting: {0:.3f}, {1:.3f}, {2:.3f}'.format(Afit, Bfit, Cfit)
        # Kfit = poly2(r, Afit, Bfit, Cfit)
        # fig, ax = plt.subplots(nrows=2)
        # ax[0].plot(r, K, 'b', label='Numerical')
        # ax[0].plot(r, Kfit, 'r', label='Fitting')
        # leg0 = ax[0].legend()
        # leg0.draggable()
        #
        # # g(r) computation
        # Na = sum([fibre.Nf for fibre in myMicrostructure.sq])/(L*H)
        # #
        # dK = np.diff(K)/np.diff(r)
        # g = dK / (Na*2*np.pi*r[:-1])
        # ax[1].plot(r[:-1], g, 'b', label='Numerical-Step')
        #
        # leg1 = ax[1].legend()
        # leg1.draggable()
        #
        plt.show()

    elif 0: # Carry out statistical analysis
        print 'First Analysis:'
        os.chdir(directory)
        statisticGeneration(L=60., H=60., num=2, filename=r'POTrsp_60x60_Vf60', algorithm=POTENTIAL,
                        directory=r'C:\Users\miguel.herraez\Desktop\VIPPER project\_DONE potential dispersion\Analysis',
                        fibersets=[{'Geometry': CIRCULAR, 'Phi': (0.0, 0.0), 'Parameters': [], 'Vf': 60., 'df': (5., 0.0), 'Material': u'CF-AS4'}],)

        # print 'Second Analysis:'
        # statisticGeneration(L=60., H=60., num=5, filename=r'POTr_60x60_Vf60_lob3', algorithm=POTENTIAL,
        #                 directory=r'C:\Users\miguel.herraez\Desktop\VIPPER project\_DONE potential dispersion\Analysis',
        #                 fibersets=[{'Geometry': LOBULAR, 'Phi': (0.0, 90.0), 'Parameters': [2,], 'Vf': 60., 'df': (5., 0.0), 'Material': u'CF-AS4'}],)

        plt.show()

        # print 'Third Analysis:'
        # statisticGeneration(L=300., H=300., num=5, filename=r'POT_300x300_Vf60', algorithm=POTENTIAL,
        #                 directory=r'C:\Users\miguel.herraez\Desktop\VIPPER project\_DONE potential dispersion\Analysis',
        #                 fibersets=[{'Geometry': CIRCULAR, 'Phi': (0.0, 0.0), 'Parameters': [], 'Vf': 60., 'df': (5., 0.0), 'Material': u'CF-AS4'}],)

        # statisticGeneration(L=150., H=150., num=5, filename=r'POT_150x150_Vf65_lob2', algorithm=POTENTIAL,
        #                 directory=r'C:\Users\miguel.herraez\Desktop\VIPPER project\_DONE potential dispersion\Analysis',
        #                 fibersets=[{'Geometry': LOBULAR, 'Phi': (0.0, 90.0), 'Parameters': [2,], 'Vf': 65., 'df': (5., 0.0), 'Material': u'CF-AS4'}],)

    elif 0: # Test potential generation
        rve_size = (50.0, 50.0)
        # Define fibersets
        fibersets = [{'Geometry': u'Circular',
                      'Phi': (0.0, 0.0),
                      'Parameters': [],
                      'Vf': 65.0,
                      'df': (7.2, 0.),
                      'Material': u'CF-AS4'},]
        tinit = time.time()
        filename = r'test_potential'
        myMicrostructure = Microstructure(rve_size=rve_size, fibersets=fibersets, gen_algorithm=POTENTIAL,
                                          tolerance=0.25)
        print 'Time: %5.1f s.' % (time.time() - tinit)

        myMicrostructure.save_rve(directory=directory, filename=filename)
        myMicrostructure.plot_rve(numbering=True)

        #

    elif 0: # Control tests for potential generation

        from Potentials import d_potential, potential
        from conjugate_gradient_functions import nonlinear_conjugate_gradient
        from myClasses import Dispersion
        from testsPG import cases

        test_id = ''

        if test_id:
            particles, (L, H) = cases(test_id)
            N = len(particles)
            dispersion = Dispersion(L, H, N)
            for p_kw in particles:
                dispersion.setParticle(**p_kw)
        else:
            L, H = 58., 58.
            Lf = 7.2
            N = 43
            phi_ref = 10.0
            dispersion = Dispersion(L, H, N)
            kw = {'shape':'CIRCULAR'}

            dispersion.setParticle(Lf, 0.0, 0.0, phi=phi_ref, fixed=True, **kw)  # Particle at the origin is fixed

            for i in range(1, N):
                x0 = 0.5*Lf + (L-Lf)*np.random.rand()
                y0 = 0.5*Lf + (H-Lf)*np.random.rand()
                d = Lf # np.random.normal(loc=Lf, scale=Lf*0.0)
                phi = phi_ref #np.random.normal(loc=phi_ref, scale=10.)
                dispersion.setParticle(d, x0, y0, phi=phi, **kw)

        dispersion.update()
        print dispersion
        dispersion.plot(title=True, numbering=True, ion=True)

        t0 = time.time()
        e = nonlinear_conjugate_gradient(d_potential, potential, dispersion, 1e-9, plot=True)
        print 'Time: {0:.1f} s'.format(time.time()-t0)

        dispersion.plot(title=True, numbering=True, margin=0.0, arrows=True)

    elif 0: # Periodic micros with lob2

        d_eff = 7.59
        Lf = fiber_shape('LOBULAR', [2,], d_eff)
        Af = np.pi/4*d_eff**2
        print 'Lf = {0:.2f}. Af = {1:.2f}'.format(Lf, Af)

        # Ar = 2.  # aspect ratio of the fibre
        L, H = 34., 34. #300., 100.  # RVE size
        Vf = 0.16369
        N = int(np.ceil(Vf*L*H/Af))
        print 'Required fibres: {0:d}. Vf = {1:.1f} %'.format(N, N*Af/L/H*100.)

        ##################################
        #### Puzzle arrangement
        ##################################
        nx = 2*int(np.sqrt(N*L/H))
        l_min = Lf/4*(np.sqrt(3)+1)
        l = L/nx
        # ny = N/nx
        ny = int(nx*H/L)
        print 'l = {0:.2f}. l_min = {1:.2f}'.format(l, l_min)
        # print nx, ny, N
        print 'nx x ny = N\n{0:d} x {1:d} = {2:d}'.format(nx, ny, nx*ny)
        print 'Final Vf = {0:.1f} %'.format(nx*ny*Af/L/H*100.)

        # Check horizontal spacing
        # if nx >= L_eff / lx:
        #     nx = int(L_eff / lx)
        # l1 = L_eff / nx  # l1 > dx
        #
        # # Check vertical spacing
        # if ny >= H_eff / ly:
        #     ny = int(H_eff / ly)
        # l2 = H_eff / ny  # l2 > dy

        # Create unit cell (4 fibres (9))
        L_, H_ = 2*L/nx, 2*H/ny
        print 'Unit cell: {0:.1f} x {1:.1f}'.format(L_,H_)
        microUnit = Microstructure(rve_size=(L_, H_), tolerance=0.01)

        microUnit.sq.append(Fiber(geometry='LOBULAR', parameters=[2,], L=Lf, phi=0.,  center=(   0,   0), period=5, Nf=1))
        microUnit.sq.append(Fiber(geometry='LOBULAR', parameters=[2,], L=Lf, phi=0.,  center=(   0,  H_), period=6, Nf=0))
        microUnit.sq.append(Fiber(geometry='LOBULAR', parameters=[2,], L=Lf, phi=0.,  center=(  L_,  H_), period=7, Nf=0))
        microUnit.sq.append(Fiber(geometry='LOBULAR', parameters=[2,], L=Lf, phi=0.,  center=(  L_,   0), period=8, Nf=0))
        microUnit.sq.append(Fiber(geometry='LOBULAR', parameters=[2,], L=Lf, phi=90., center=(L_/2,   0), period=4, Nf=1))
        microUnit.sq.append(Fiber(geometry='LOBULAR', parameters=[2,], L=Lf, phi=90., center=(L_/2,  H_), period=2, Nf=0))
        microUnit.sq.append(Fiber(geometry='LOBULAR', parameters=[2,], L=Lf, phi=90., center=(   0,H_/2), period=1, Nf=1))
        microUnit.sq.append(Fiber(geometry='LOBULAR', parameters=[2,], L=Lf, phi=90., center=(L_  ,H_/2), period=3, Nf=0))
        microUnit.sq.append(Fiber(geometry='LOBULAR', parameters=[2,], L=Lf, phi=0.,  center=(L_/2,H_/2), period=0, Nf=1))
        # microUnit.plot_rve(mute=False, show_plot=False)

        micro = patternedMicrostructure(microUnit, ncols=nx/2, nrows=ny/2, fileout='puzzleT',
                                        directory='.', save=True, plot=False)

        micro.plot_rve(mute=False)

    elif 0: # Periodic microstructure with non-circular fibers
        d_eff = 7.3
        phi = 0.
        Lf = fiber_shape(CSHAPE, [0.5, 120], d_eff)
        Af = np.pi/4*d_eff**2
        print 'Lf = {0:.2f}. Af = {1:.2f}'.format(Lf, Af)

        # S0 = 6., 22., #100., 300.  # RVE size
        S0 = 300., 100.
        Vf = 0.65
        fiber = Fiber(geometry=CSHAPE, parameters=[0.5, 120], L=Lf, phi=phi, aux=True)

        # print zip(fiber.vertices[0],fiber.vertices[1])

        centers = periodic_layout(S0, Vf, fiber, pattern=HEXAGONAL, tolerance=0.01,
                                  optPeriod=FULL)

        micro = Microstructure(rve_size=S0, tolerance=0.01)

        for i, row in enumerate(centers):
            for j, center in enumerate(row):
                micro.sq.append(Fiber(geometry=fiber.geometry, parameters=fiber.parameters, L=Lf, phi=phi,
                                      center=center, period=0, Nf=1))

        # micro.setPeriodicity()
        micro.save_rve(directory=directory, filename='cshape')
        micro.plot_rve(mute=False, numbering=True)

    elif 0: # Fiber vertices

        fiber = Fiber(geometry=CSHAPE, parameters=[0.5, 120], L=1.0, phi=-45., aux=True)

        print zip(fiber.vertices[0],fiber.vertices[1])

    elif 0: # S_2 statistical function
        # Simplify microstructure to b/w grid
        filename = r'59'
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename))

        myMicrostructure.S2_function(pixelSize=0.5, spots=False, fibres=False)

        # print 'Bounds: {0:.1f}% < Vf < {1:.1f}%'.format(np.min(myLvf)*100, np.max(myLvf)*100)
        # plt.ion()
        plt.show()

    elif 0: # Test closest fibre and orientation
        directory = r'C:\Users\miguel.herraez\Desktop\VIPPER project\_DONE potential dispersion\Analysis'
        os.chdir(directory)
        # filename = r'POTrsp_60x60_Vf60_003'
        filename = r'POTr_100x100_Vf60_001'
        # filename = r'POT_300x300_Vf65_005'
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename))
        # myMicrostructure.plot_rve(filename=filename, directory=directory, numbering=True, save=True,
        #                           matrix_colour='', fibre_colour='', show_plot=True)
        # refFiber = 13
        # reference = myMicrostructure.sq[refFiber-1]
        # # First neighbour
        # found_1 = myMicrostructure.findClosest(reference, f=1)
        # print 'First neighbour of {0} is {1}'.format(refFiber, found_1.id+1)
        # # Second neighbour
        # found_2 = myMicrostructure.findClosest(reference, f=2)
        # print 'Second neighbour of {0} is {1}'.format(refFiber, found_2.id+1)
        #
        # # First neighbour
        # phi_1, f1 = myMicrostructure.orientationClosest(reference, f=1)
        # print 'First neighbour of {0} is {1}, {2}'.format(refFiber, found_1.id+1, phi_1)
        # # Second neighbour
        # phi_2, f2 = myMicrostructure.orientationClosest(reference, f=2)
        # print 'Second neighbour of {0} is {1}, {2}'.format(refFiber, found_2.id+1, phi_2)

        myMicrostructure.orientationsDistribution(plot=True)
        ##

    elif 0: # Test pixel grid raster

        filename = r'box2d_after'
        # filename = '01_box2d'
        myMicrostructure = Microstructure(read_microstructure=os.path.join(directory, filename))
        # myMicrostructure.easy_plot(filename='simple_plot')
        # nx < 3000
        myMicrostructure.pixelGrid2(3000, plot=False, spots=False, save='pixelgrid_3000x3000.txt', javi=True)
        #

    else:
        print 'Enable one option...'
        pass
        #

#######################################################################################

if __name__ == '__main__':
    print 'This is a test of this module'
    main()


#--End of file---------------------------------------------------------------------
