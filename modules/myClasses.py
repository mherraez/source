__author__ = 'miguel.herraez'

#------ IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from matplotlib.patches import Polygon as mplPolygon
from FibresGeom import fiber_shape, coord2vert

from shapely.geometry import Polygon

try:  # Enable speedups from GEOS library
    from shapely import speedups

    if speedups.available:
        speedups.enable()
except:
    print 'Could not speedup Shapely module'


#############################################################
#------ CLASSES

# Dispersion of particles
class Dispersion():
    ind = 0
    Particles = {}
    Nm = 0
    tolerance = 0.

    def __init__(self, L, H, N):
        self.L = L
        self.H = H
        self.N = N

    def __str__(self):
        p = Dispersion.Particles
        s = '{0:>6s} {1:>6s} {2:>6s} {3:>6s} {4:>9s} {5:>9s} {6:>6s} {7:<20s} {8:>6s} {9:>6s}\n'.format('Index', 'x0', 'y0', 'd',
                                            'shape', 'periodic', 'slave', 'neighbours', 'p_x', 'p_y')
        for k in sorted(Dispersion.Particles.keys()):
            if p[k].periodic == None:
                per = 'NA'
            else:
                per = str(p[k].periodic)
            s += '{0:6d} {1:6.2f} {2:6.2f} {3:6.2f} {4:>9s} {5:>9} {6:>6s} {7:<20} {8:>+6.3f} {9:>+6.3f}\n'.format(p[k].ind, p[k].center[0],
                                            p[k].center[1], p[k].L, p[k].shape, per, str(p[k].slave), p[k].neighbours,
                                            -p[k].dV[0], -p[k].dV[1])
        return s

    def setParticle(self, d=0., x0=0., y0=0., phi=0.0, shape='CIRCULAR', parameters=[], auxiliary=False, fixed=False, periodic=None,
                    slave=False, material='CF-AS4', L=None):
        """
        :param d: diameter
        :param x0: center x
        :param y0: center y
        Creates a circular particle
        """
        p = Particle(d, x0, y0, phi, shape, parameters, auxiliary, fixed, periodic, Dispersion.ind, slave, material, L)
        self.Particles[Dispersion.ind] = p
        Dispersion.ind += 1
        if not slave: Dispersion.Nm += 1
        return p

    def getCentres(self):
        """
        :return List of centres (x1, y1, x2, y2, x3, y3...)
        """
        centres = []
        keys = Dispersion.Particles.keys()
        p = Dispersion.Particles
        for k in sorted(keys):
            centres += list(p[k].center)
        return centres

    def setCentres(self, xi, dm):
        """
        :param xi: Original centers of the particles as: (x0, y0, x1, y1, x2, y2...)
        :param dm: Displacement vector of master particles (u0, v0, u1, v1, u2, v2...))
        """
        # keys = Dispersion.Particles.keys()
        # print Dispersion.Nm
        p = Dispersion.Particles
        # [p[k].set_center(x[2*i:2*i+2]) for i,k in enumerate(sorted(keys))]
        xf = [xi[i]+dm[i] for i in xrange(2*Dispersion.Nm)]
        # print xf
        [p[i].set_center(xf[2*i:2*i+2]) for i in xrange(Dispersion.Nm)]
        return 0

    def getNeighbours(self, particle, margin=1.0):
        """
        Updates the list of particles neighbouring particle with the given 'index'
        :param index: Particle ind
        """
        particle.neighbours = [k for k,p in Dispersion.Particles.items() if (p != particle)
                         and (self._closeParticles(particle,p, margin))]

    def update(self, neighbours=True, periodic=True):

        # Update the periodic particles in case there are
        if periodic:
            if np.all([not p.auxiliary for p in self.Particles.values()]):
                # Set periodics
                self.setPeriodics()
            else:
                # Update periodics
                self.updatePeriodics()

        # Updates neighbours of all particles
        if neighbours:
            [self.getNeighbours(p) for p in self.Particles.values()]

    def setPeriodics(self):
        # Create initial periodic particles

        # Check possible periodic particles
        parts = [p for p in self.Particles.values() if (p.center[0]-0.5*p.L < 0.0)    or (p.center[1]-0.5*p.L < 0.0) or
                                                       (p.center[0]+0.5*p.L > self.L) or (p.center[1]+0.5*p.L > self.H) ]

        for p in parts:
            left, top, right, bottom = False, False, False, False
            if np.any(p.vertices[0, :] < 0.0):
                # per = 1
                left = True
                newP = self.setParticle(p.d, p.center[0]+self.L, p.center[1], phi=p.phi, shape=p.shape, parameters=p.parameters,
                                        auxiliary=True, fixed=p.fixed, periodic=p.ind, slave=True)
                p.periodic = newP.ind
            elif np.any(p.vertices[0, :] > self.L):
                # per = 3
                right = True
                newP = self.setParticle(p.d, p.center[0]-self.L, p.center[1], phi=p.phi, shape=p.shape, parameters=p.parameters,
                                        auxiliary=True, fixed=p.fixed, periodic=p.ind, slave=True)
                p.periodic = newP.ind

            if np.any(p.vertices[1, :] > self.H):
                # per = 2
                top = True
                newP = self.setParticle(p.d, p.center[0], p.center[1]-self.H, phi=p.phi, shape=p.shape, parameters=p.parameters,
                                        auxiliary=True, fixed=p.fixed, periodic=p.ind, slave=True)
                p.periodic = newP.ind
            elif np.any(p.vertices[1, :] < 0.0):
                # per = 4
                bottom = True
                newP = self.setParticle(p.d, p.center[0], p.center[1]+self.H, phi=p.phi, shape=p.shape, parameters=p.parameters,
                                        auxiliary=True, fixed=p.fixed, periodic=p.ind, slave=True)
                p.periodic = newP.ind

            if left and bottom: # Lower-left corner
                # per = 5
                newP = self.setParticle(p.d, p.center[0]+self.L, p.center[1]+self.H, phi=p.phi, shape=p.shape, parameters=p.parameters,
                                        auxiliary=True, fixed=p.fixed, periodic=p.ind, slave=True)
                p.periodic = newP.ind
            elif left and top: # Upper-left corner
                # per = 6
                newP = self.setParticle(p.d, p.center[0]+self.L, p.center[1]-self.H, phi=p.phi, shape=p.shape, parameters=p.parameters,
                                        auxiliary=True, fixed=p.fixed, periodic=p.ind, slave=True)
                p.periodic = newP.ind
            elif right and top: # Upper-right corner
                # per = 7
                newP = self.setParticle(p.d, p.center[0]-self.L, p.center[1]-self.H, phi=p.phi, shape=p.shape, parameters=p.parameters,
                                        auxiliary=True, fixed=p.fixed, periodic=p.ind, slave=True)
                p.periodic = newP.ind
            elif right and bottom: # Lower-right corner
                # per = 8
                newP = self.setParticle(p.d, p.center[0]-self.L, p.center[1]+self.H, phi=p.phi, shape=p.shape, parameters=p.parameters,
                                        auxiliary=True, fixed=p.fixed, periodic=p.ind, slave=True)
                p.periodic = newP.ind

        return 0

    def updatePeriodics(self):

        # Update position of periodic fibres and delete those that are completely outside the dominion
        p = Dispersion.Particles

        # 1. Check particles that became not-periodic (remove and update)
        p_per = [pp for pp in p.values() if (pp.periodic != None)] # and (not pp.slave)]
        p_p = [pp for pp in p_per if np.all(pp.vertices[0, :] < 0.0) or np.all(pp.vertices[0, :] > self.L) \
               or np.all(pp.vertices[1, :] > self.H) or np.all(pp.vertices[1, :] < 0.0)]
        for pp in p_p:

            if not pp.slave:  # Master particle exits
                # print 'Delete Master {0}. Slave {1}'.format(pp.ind, p[pp.periodic].ind)
                newCenter = p[pp.periodic].center
                pp.set_center(newCenter)
                p[pp.periodic].remove()
                pp.periodic = None
                pp.auxiliary = None
            else:
                # print 'Delete Slave {0}. Master {1}'.format(pp.ind, p[pp.periodic].ind)
                p[pp.periodic].periodic = None
                pp.remove()

            # print 'To delete particle', pp.ind

        # 2. Check particles that have become periodic (duplicate)
        p_np = [pp for pp in p.values() if pp.periodic == None]
        for pn in p_np:
            if np.any(pn.vertices[0, :] < 0.0):
                newP = self.setParticle(pn.d, pn.center[0]+self.L, pn.center[1], phi=pn.phi, shape=pn.shape, parameters=pn.parameters,
                                        auxiliary=True, fixed=pn.fixed, periodic=pn.ind, slave=True, material=pn.material)
                pn.periodic = newP.ind
                # print 'New periodic {0}-{1}'.format(pn.ind, newP.ind)
            elif np.any(pn.vertices[0, :] > self.L):
                newP = self.setParticle(pn.d, pn.center[0]-self.L, pn.center[1], phi=pn.phi, shape=pn.shape, parameters=pn.parameters,
                                        auxiliary=True, fixed=pn.fixed, periodic=pn.ind, slave=True, material=pn.material)
                pn.periodic = newP.ind
                # print 'New periodic {0}-{1}'.format(pn.ind, newP.ind)
            elif np.any(pn.vertices[1, :] > self.H):
                newP = self.setParticle(pn.d, pn.center[0], pn.center[1]-self.H, phi=pn.phi, shape=pn.shape, parameters=pn.parameters,
                                        auxiliary=True, fixed=pn.fixed, periodic=pn.ind, slave=True, material=pn.material)
                pn.periodic = newP.ind
                # print 'New periodic {0}-{1}'.format(pn.ind, newP.ind)
            elif np.any(pn.vertices[1, :] < 0.0):
                newP = self.setParticle(pn.d, pn.center[0], pn.center[1]+self.H, phi=pn.phi, shape=pn.shape, parameters=pn.parameters,
                                        auxiliary=True, fixed=pn.fixed, periodic=pn.ind, slave=True, material=pn.material)
                pn.periodic = newP.ind
                # print 'New periodic {0}-{1}'.format(pn.ind, newP.ind)

        return 0

    def updateDisp(self, pm):

        p = self.Particles

        for i in xrange(Dispersion.Nm):
            p[i].pm = pm[2*i:2*i+2]
            # Check periodic
            if p[i].periodic:
                p[p[i].periodic].pm = pm[2*i:2*i+2]

    def _closeParticles(self, p1, p2, margin=0.0):
        distance = np.sqrt( (p1.center[0] - p2.center[0])**2 + (p1.center[1] - p2.center[1])**2 )
        if distance < 0.5*(p1.L+p2.L)+margin:
            return True
        else:
            return False

    def volumeFraction(self):
        """
        :return: Particles volume fraction (over 1)
        """
        area = self.L*self.H
        areaFibre = sum([p.area for p in Dispersion.Particles.values() if not p.auxiliary])
        # areaFibre = 0.0
        # for p in Dispersion.Particles.values():
        #     if not p.auxiliary:
        #         areaFibre += p.area

        return areaFibre/area

    def plot(self, numbering=False, title=True, ion=False, margin=0.0, arrows=True, save=False, show=True,
             ticks=True):

        if ion:
            plt.ion()
        else:
            plt.close('all')
            plt.ioff()

        maxDimension = float(max(self.L, self.H))
        lmax = 8.0
        sizex = self.L / maxDimension * lmax
        sizey = self.H / maxDimension * lmax
        fig, ax = plt.subplots(num=None, figsize=(sizex, sizey), dpi=100, facecolor='w', edgecolor='k')

        ax.set_xlim((-margin, self.L+margin))
        ax.set_ylim((-margin, self.H+margin))
        ax.set_aspect('equal')
        if ticks:
            ax.set_xticks([0, self.L])
            ax.set_yticks([0, self.H])
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if title:
            ax.set_title('$V_f=$ {0:.1f}% $N=${1:d}'.format(self.volumeFraction()*100., self.N))

        for p in Dispersion.Particles.values():
            color = 'g'
            if p.auxiliary: color = 'y'
            hatch = ''
            if p.fixed: hatch = '//'
            if p.shape.upper() == 'CIRCULAR':
                fiberpoly = patches.Circle(p.center, radius=(p.L-Dispersion.tolerance)*0.5, facecolor=color,
                                           edgecolor='k', hatch=hatch, alpha=0.5)
            else:
                # Scale considering tolerance
                p_star = p.copy()
                L_star = (p.d-Dispersion.tolerance)/p.d * p.L
                p_star.set_L(L=L_star)
                polyvert = np.asarray(p_star.polygonly.exterior)
                fiberpoly = patches.Polygon(polyvert, facecolor=color, edgecolor='k', hatch=hatch, alpha=0.5)

            ax.add_patch(fiberpoly)
            if numbering:
                ax.text(p.center[0], p.center[1], p.ind, color='k', verticalalignment='center',
                        horizontalalignment='center', weight='semibold')
            if arrows and not p.fixed:
                ax.arrow(p.center[0], p.center[1], -p.dV[0], -p.dV[1], head_width=0.10, head_length=0.20, fc='r', ec='r')
                ax.arrow(p.center[0], p.center[1], +p.pm[0], +p.pm[1], head_width=0.15, head_length=0.25, fc='b', ec='b')

        ax.add_patch(patches.Rectangle( (0,0) , self.L, self.H, color='k', fill=False, linewidth=1.5))
        plt.tight_layout()
        if show:
            plt.show()

        if save:
            fig.savefig(save)

    def updatingPlot(self, ax, pause=0.001, margin=[1,1]):
        # ax.cla()
        for p in self.Particles.values():
            if p.auxiliary: ax.plot(p.center[0], p.center[1], 'g+')
            else: ax.plot(p.center[0], p.center[1], 'b+')
        # plt.draw()
        ax.set_xlim([-1,self.L+margin[0]])
        ax.set_ylim([-1,self.H+margin[1]])
        ax.add_patch(patches.Rectangle( (0,0) , self.L, self.H, color='r', fill=False))
        plt.pause(pause)
        return 0

    @staticmethod
    def resetList():
        Dispersion.Particles = {}
        Dispersion.ind = 0
        Dispersion.Nm = 0


# Particle class
class Particle():

    def __init__(self, d=1., x0=0.0, y0=0.0, phi=0.0, shape='CIRCULAR', parameters=[], auxiliary=False, fixed=False, periodic=None,
                 ind=None, slave=False, material='CF-AS4', L=None):

        self.factor = fiber_shape(shape, parameters, 1.0)

        if L:
            self.L = L
            self.d = L / self.factor
        elif d:
            self.d = d              # Equivalent diameter (circle with the same area)
            self.L = self.factor * d     # Characteristic length (diameter of the circumscribing circle)
        else:
            raise KeyError('Either real diameter (L) or equivalent diameter (d) needs to be specified.')

        self.ind = ind               # Unique ID for each particle
        self.center = (x0, y0)       # Center of the circumscribing circle
        self.phi = phi               # Misalignment angle
        self.convex = True           # TODO Convex polygons permit cheaper collision algorithms (e.g. separating axis)**
        self.shape = shape           # Shape of the particle
        self.parameters = parameters # Additional parameters required for some particle geometries
        self.neighbours = []         # Neighbour particles list

        self.auxiliary = auxiliary   # Auxiliary particles represent virtual periodic particles (slave particle)
        self.fixed = fixed           # Particle is unmovable
        self.periodic = periodic     # 'ind' of the periodic particle
        self.slave = slave           # slave particles move according to their periodic master particle
        self.material = material     # Material the particle is made of (unused)

        # Array with the vertices of the particle
        self.vertices = coord2vert(geometry=shape, L=self.L, center=(x0,y0), phi=phi, parameters=parameters)
        self.set_polygonly()
        self.area = np.pi*0.25*self.d**2  # Area of the particle
        # self.area = self.polygonly.area

        # Potential variables
        self.V = 0.0            # Potential
        self.dV = [0.0, 0.0]    # Potential gradient
        self.pm = [0.0, 0.0]    # Displacement vector

    def translate(self, dx, dy):
        if not self.fixed:
            self.center = (self.center[0]+dx, self.center[1]+dy)
            self.vertices[0] += dx
            self.vertices[1] += dy
            self.set_polygonly()

    def set_center(self, center):
        if not self.fixed and not self.slave:
            dx = center[0] - self.center[0]
            dy = center[1] - self.center[1]
            self.vertices[0] += dx
            self.vertices[1] += dy
            self.center = center
            self.set_polygonly()
            if self.periodic != None:
                per = Dispersion.Particles[self.periodic]
                per.vertices[0] += dx
                per.vertices[1] += dy
                per.center = (per.center[0]+dx, per.center[1]+dy)
                per.set_polygonly()

    def set_L(self, L=None, increment=0.0):
        if L:
            self.L = L
        elif increment:
            self.L += increment
        else:
            print 'Size of particle {} was not modified'.format(self.ind)

        self.d = self.L / self.factor

        self.vertices = coord2vert(geometry=self.shape, L=self.L, center=self.center, phi=self.phi, parameters=self.parameters)
        self.set_polygonly()
        self.area = np.pi*0.25*self.d**2  # Area of the particle
        # self.area = self.polygonly.area   # Area of the particle

    def rotate(self, dphi):
        if not self.fixed:
            self.phi += dphi
            # TODO rotate polygon (vertices)
            self.set_polygonly()

    def set_polygonly(self):
        """Returns a shapely Polygon object"""
        self.polygonly = Polygon(zip(self.vertices[0], self.vertices[1]))

    def collision(self, other):
        """
        Determine if a Particle intersects another Particle (True or False)
        :param other: Particle
        :return: if the pair of particles intersects
        """
        if self.shape.upper() == other.shape.upper() == 'CIRCULAR':
            if distancePoints(self.center, other.center) < 0.5*(self.d+other.d):
                return True
            else:
                return False

        return self.polygonly.intersects(other.polygonly)

    def remove(self):
        # print 'Delete particle', self.ind
        del Dispersion.Particles[self.ind]

    def copy(self):
        return Particle(self.d, self.center[0], self.center[1], phi=self.phi, shape=self.shape,
                        parameters=self.parameters, material=self.material)


def distancePoints(c1, c2):

    return np.sqrt( (c1[0] - c2[0])*(c1[0] - c2[0]) + (c1[1] - c2[1])*(c1[1] - c2[1]) )


#############################################################
# TEST PROGRAM
if __name__ == '__main__':
    from Potentials import d_potential, potential
    N = 200
    d_ref = 2.0
    phi_ref = 15.0
    L, H = 30., 10.
    dispersion = Dispersion(L,H,N)

    # kw = {'shape':'LOBULAR', 'parameters':[5,]}
    # kw = {'shape':'SPOLYGONAL', 'parameters':[5, 0.2]}
    # kw = {'shape':'POLYGONAL', 'parameters':[4,]}
    kw = {'shape':'CIRCULAR'}

    dispersion.setParticle(d_ref, 0.0, 0.0, phi=phi_ref, fixed=True)  # Particle at the origin is fixed

    import testsPG
    test = 2
    particles, (L, H) = testsPG.cases(test)
    N = len(particles)
    dispersion = Dispersion(L, H, N)
    for p_kw in particles:
        dispersion.setParticle(**p_kw)

    # Update neighbours and initial periodic particles
    dispersion.update()
    print dispersion

    ener = potential(dispersion)
    print ener
    dener = d_potential(dispersion)
    # print dener

    # print dispersion
    dispersion.plot(title=True, numbering=True, ion=False, margin=1.0, arrows=True)

