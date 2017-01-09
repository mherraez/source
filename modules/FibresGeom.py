__author__ = 'miguel.herraez'

#----- IMPORTS
import numpy as np
import os
from shapely.geometry import Polygon, Point
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon

#from matplotlib.collections import PatchCollection
#import matplotlib.pyplot as plt

try:  # Enable speedups from GEOS library
    from shapely import speedups

    if speedups.available:
        speedups.enable()
except:
    pass

#----- END OF IMPORTS

#----- CONSTANTS

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

# Compaction algorithm
POINT = 'POINT'
DIRECTION = 'DIRECTION'
STIRRING = 'STIRRING'
BACKWARD = 'BACKWARD'  # Outward direction
FORWARD = 'FORWARD'  # Inward direction
BOX2D = 'BOX2D'         # Physics based compaction (gravity, shaking)

# Periodicity
FULL = 'FULL'       # Full periodicity
HORIZ = 'HORIZ'     # Not horizontally periodic
VERT = 'VERT'       # Not vertically periodic
NONE = 'NONE'       # Not periodic
# Forbidden periods
periodicity = {FULL: [],
               HORIZ: [1, 3, 5, 6, 7, 8],
               VERT: [2, 4, 5, 6, 7, 8],
               NONE: [1, 2, 3, 4, 5, 6, 7, 8]}

#----- END OF CONSTANTS DEFINITION

#----- EXCEPTIONS
class ShapeNotAvailable(Exception): pass

#----- METHODS

def fiber_shape(geometry, parameters, df):
    """ Sets homogeneous equivalence for different shapes to match equal cross section """

    if geometry.upper() == CIRCULAR:
        Lf = df

    elif geometry.upper() == POLYGON:
        m = int(parameters[0])
        if m < 3:
            raise ValueError('Minimum number of edges is 3')
        alfa = np.pi / m
        Lf = df * np.sqrt(np.pi / (m / 2. * np.sin(2 * alfa)))

    elif geometry.upper() == SPOLYGON:
        m = int(parameters[0])
        if m < 3:
            raise ValueError('Minimum number of edges is 3')
        alfa = np.pi / m
        # chi = np.cos(alfa)*0.5
        chi = float(parameters[1])
        # aux = 0.5*m*np.sin(2*alfa) - chi**2*(m*np.tan(alfa)-np.pi)
        aux = (m * np.cos(alfa)**2) * ((1 - chi**2) * np.tan(alfa) + chi**2 * alfa)
        R_old = 0.5 * df * np.sqrt(np.pi / aux)
        Lf = 2 * R_old * (1 - chi * (1 - np.cos(alfa)))

    elif geometry.upper() == LOBULAR:
        m = int(parameters[0])
        alfa = np.pi / m
        y0 = (0.5 / (1. + 1. / np.sin(alfa)))
        aux = m * ((1. / np.tan(alfa) + np.sqrt(3)) + np.pi / m)
        Y = aux * y0 ** 2
        Lf = df * np.sqrt(np.pi / 4. / Y)

    elif geometry.upper() == ELLIPTICAL:
        e = float(parameters[0])  # Eccentricity (0,1)
        Lf = df * (1 - e * e) ** (-0.25)

    elif geometry.upper() == OVAL:
        alfa = float(parameters[0])  # Slenderness (>1)
        beta = float(parameters[1])  # Flatness (0,1)
        R2_ = beta / alfa
        R1_ = (alfa + 1 / alfa - 2 * beta) / (2 - 2 * beta)
        theta = np.arcsin((1 - R2_) / (R1_ - R2_))
        aux = R2_ ** 2 * (np.pi + 2 * theta * ((R1_ / R2_) ** 2 - 1) - np.sin(2 * theta) * (R1_ / R2_ - 1) ** 2)
        Lf = df * np.sqrt(np.pi / aux)

    elif geometry.upper() == CSHAPE:
        chi = float(parameters[0])
        theta = float(parameters[1]) * np.pi / 180.0
        aux = theta / 2 * (1 - chi ** 2) + np.pi / 4 * (1 - chi) ** 2
        Lf = df * np.sqrt(np.pi / aux)

    else:
        raise ShapeNotAvailable, 'Shape is not available'

    return Lf


def coord2vert(geometry=CIRCULAR, L=1.0, center=(0.0,0.0), phi=0.0, parameters=None):
    """Returns an array with vertices of the polygon object"""

    # Circular section
    if geometry.upper() == CIRCULAR:
        R = L / 2.0  # note phi is unnecessary
        m = 20  # Number of vertex
        theta = np.linspace(0, -360, m)
        # X_ = np.zeros([1, m])
        # Y_ = np.zeros([1, m])
        X_ = np.matrix(R * np.cos(theta * np.pi / 180))
        Y_ = np.matrix(R * np.sin(theta * np.pi / 180))

    # Regular polygon section
    elif geometry.upper() == POLYGON:
        m = int(parameters[0])
        R = L * 0.5  # Circumscribed circumference
        alfa = np.pi / m
        theta = alfa + np.linspace(0.0, -2 * np.pi, m + 1)
        X_ = R * np.cos(theta)
        Y_ = R * np.sin(theta)

    # Smoothed regular polygon
    elif geometry.upper() == SPOLYGON:
        m = int(parameters[0])
        chi = float(parameters[1])
        alfa = np.pi / m
        R_new = L * 0.5  # Circumscribed circumference of smoothed polygon
        R_old = R_new / (1 - chi * (1 - np.cos(alfa)))  # Circumscribed circumference of sharp polygon
        r = chi * np.cos(alfa) * R_old  # Smoothing radius
        # R = r*np.cos(alfa)*0.5
        yo = R_old * np.cos(alfa) - r
        xo = yo * np.tan(alfa)
        m1 = max(np.ceil(180.0 * alfa / 5 / np.pi), 2)
        theta = np.linspace(0, 2 * alfa, num=m1)
        X_ = xo + r * np.sin(theta)
        Y_ = yo + r * np.cos(theta)
        X_ = np.append(0, X_)
        Y_ = np.append(R_old * np.cos(alfa), Y_)
        X1 = X_
        Y1 = Y_
        beta = -2 * np.pi / m * np.linspace(1, m - 1, num=m - 1)
        for i in range(m - 1):
            s = np.sin(beta[i])
            c = np.cos(beta[i])
            T = np.matrix([[c, -s], [s, c]])
            A = T * np.matrix([X1, Y1])
            xaux = np.asarray(A[0, :])
            yaux = np.asarray(A[1, :])
            X_ = np.append(X_, xaux)
            Y_ = np.append(Y_, yaux)

    # Lobular
    elif geometry.upper() == LOBULAR:
        m = int(parameters[0])
        alfa = np.pi / m
        R = L * 0.5 / (1 + 1 / np.sin(alfa))
        #            R = L/(4/3*np.sin(alfa) + 1/np.cos(alfa))*(m % 2 == 1) + (L*0.5/(1+1/np.sin(alfa)))*(m % 2 == 0)
        r = R / np.sin(alfa)
        r_ = R * (1 / np.tan(alfa) + np.sqrt(3))
        gamma = np.arcsin(0.5)
        N1 = np.ceil(gamma * 180 / 10 / np.pi)
        N2 = np.ceil((alfa + gamma) * 180 / 10 / np.pi)
        ta = np.linspace(np.pi + alfa, np.pi + alfa + gamma, N1)
        tb = np.linspace(alfa + gamma, 0, N2)
        xa = r_ * np.cos(alfa) + R * np.cos(ta)
        ya = r_ * np.sin(alfa) + R * np.sin(ta)
        xb = r + R * np.cos(tb[1:])
        yb = R * np.sin(tb[1:])
        x1 = np.append(xa, xb)
        y1 = np.append(ya, yb)
        X_ = np.append(x1, x1[-2:0:-1])
        Y_ = np.append(y1, -y1[-2:0:-1])
        X1 = X_
        Y1 = Y_
        beta = -2 * alfa * np.linspace(1, m - 1, num=m - 1)
        for i in range(m - 1):
            s = np.sin(beta[i])
            c = np.cos(beta[i])
            T = np.matrix([[c, -s], [s, c]])
            A = T * np.matrix([X1, Y1])
            xaux = np.asarray(A[0, :])
            yaux = np.asarray(A[1, :])
            X_ = np.append(X_, xaux)
            Y_ = np.append(Y_, yaux)

    # Ellipsis
    elif geometry.upper() == ELLIPTICAL:
        e = float(parameters[0])  # Eccentricity
        a = L * 0.5
        b = a * np.sqrt(1 - e * e)
        theta = np.linspace(0.0, -2 * np.pi, 128)
        r = b / np.sqrt(1 - (e * np.cos(theta)) ** 2)
        X_ = r * np.cos(theta)
        Y_ = r * np.sin(theta)

    # Oval
    elif geometry.upper() == OVAL:
        alfa = float(parameters[0])
        beta = float(parameters[1])
        a = L * 0.5
        b = a / alfa
        R2_ = beta / alfa
        R2 = a * R2_
        R1_ = (alfa + 1 / alfa - 2 * beta) / (2 - 2 * beta)
        R1 = a * R1_
        theta0 = np.arcsin((1 - R2_) / (R1_ - R2_))
        phi0 = np.pi * 0.5 - theta0
        theta = np.linspace(0, theta0, 15)[:-1]  # Exclude last point
        phi = np.linspace(theta0, np.pi * 0.5, 10)  # Exclude last point
        xth = R1 * np.sin(theta)
        xphi = R1 * np.sin(theta0) + R2 * (np.sin(phi) - np.cos(phi0))
        x1 = np.append(xth, xphi)
        yth = R1 * np.cos(theta) - (R1 - b)
        yphi = R2 * np.cos(phi)
        y1 = np.append(yth, yphi)
        # Symmetry
        X1 = np.append(x1, x1[-2:0:-1])
        Y1 = np.append(y1, -y1[-2:0:-1])
        # Rotation
        X_ = np.append(X1, -X1)
        Y_ = np.append(Y1, -Y1)
        #plt.plot(X_,Y_,'ko-')

    # C-Shaped
    elif geometry.upper() == CSHAPE:
        chi = float(parameters[0])
        theta = float(parameters[1]) * np.pi / 180.0
        Rout = L * 0.5
        Rin = Rout * chi
        r = 0.5 * (Rout - Rin)
        dx = 0.5 * (Rin + Rout)
        alfa1 = np.linspace(np.pi, np.pi - 0.5 * theta, 15)[:-1]
        x1 = dx + Rout * np.cos(alfa1)
        y1 = Rout * np.sin(alfa1)
        dx2 = dx + dx * np.cos(np.pi - 0.5 * theta)
        dy2 = dx * np.sin(np.pi - 0.5 * theta)
        alfa2 = np.linspace(np.pi - 0.5 * theta, -0.5 * theta, 8)[:-1]
        x2 = dx2 + r * np.cos(alfa2)
        y2 = dy2 + r * np.sin(alfa2)
        x = np.append(x1, x2)
        y = np.append(y1, y2)
        alfa3 = np.linspace(np.pi - 0.5 * theta, np.pi, 15)
        x3 = dx + Rin * np.cos(alfa3)
        y3 = Rin * np.sin(alfa3)
        x = np.append(x, x3)
        y = np.append(y, y3)
        # Symmetry
        X_ = np.append(x, x[-2:0:-1])
        Y_ = np.append(y, -y[-2:0:-1])
        # plt.plot(X_,Y_,'ko-')
        # plt.plot(x, y, 'bs-')
        # plt.show()
    else:
        raise TypeError('Shape is not available')
    #            print shape,'is not available'
    #             return

    # Rotation
    if geometry.upper() != CIRCULAR:  # Circles are not rotated
        s = np.sin(phi * np.pi / 180.0)
        c = np.cos(phi * np.pi / 180.0)
        T = np.matrix([[c, -s], [s, c]])
        A = T * np.matrix([X_, Y_])
        X_ = A[0, :]
        Y_ = A[1, :]

    # Translation
    X = np.asarray(X_ + center[0])
    Y = np.asarray(Y_ + center[1])

    # Output
    vertices = np.asarray([X[0], Y[0]])

    return vertices

