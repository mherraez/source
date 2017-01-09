__author__ = 'miguel.herraez'

import numpy as np

pi = np.pi


### ARCHIMEDEAN SPIRAL
class archimedeanSpiral():
    """
    Archimedean spiral
    """

    def __init__(self, a, tmax, N=None):
        """
        :param a: growth rate, such that a/(2*pi) is the distance between consecutive loops
        :param tmax: maximum angle (in radians), such that tmax/(2*pi) is the number of loops of the spiral
        :return:
        """
        self.a = a
        self.tmax = tmax
        self.length = self.lengthTheta(self.tmax)

        # Generate spiral with N points
        if not N: N = int(tmax / (2 * pi) * 100)
        self.theta = np.linspace(0, tmax, N)
        self.r, self.theta, self.x, self.y = self.angleToPoints(self.theta)

    def lengthTheta(self, theta):
        """
        :param theta: theta must be in radians
        :return: Length of the spiral (a) for a given theta angle
        """
        return 0.5 * self.a * (theta * np.sqrt(1 + theta * theta) + np.log(theta + np.sqrt(1 + theta * theta)))

    def generatePoints(self, N=100, resolution=pi / 10):
        """
        Generates an equally spaced distribution of points along the spiral
        :param N: number of points to be generated
        :param resolution: angle intervals to exactly compute spiral length
        :return:
        """
        l_spiral = self.length
        # Set control points every 'resolution'
        t_c = np.arange(0, self.tmax, resolution)
        t_c = np.append(t_c, self.tmax)
        l_c = self.lengthTheta(t_c)
        f = np.linspace(0, 0.99, N)
        newtTheta = []
        for fraction in f:
            thisLength = fraction * l_spiral
            i = np.argmax(l_c > thisLength)  # Index of the previous angle
            aveRadius = self.a * (t_c[i] + resolution / 2)
            # print thisLength, l_c[i], t_c[i]
            # print (thisLength - l_c[i])/aveRadius + t_c[i]
            newtTheta.append((thisLength - l_c[i]) / aveRadius + t_c[i])

        return newtTheta

    def angleToPoints(self, thetaList):
        """
        Computes the polar and cartesian coordinates of the points given the angles
        :param thetaList: list of angles (in radians)
        :return: A tuple of four lists: r, theta, x, y
        """
        r = self.a * np.array(thetaList)
        theta = np.array(thetaList)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return r, theta, x, y


### RECTANGULAR SPIRAL
class RectSpiral():
    def __init__(self, dx=1., dy=1.):
        self.dx = dx
        self.dy = dy
        self.pieces = []
        self.length = 0.

    def addPiece(self, newPiece):
        #		newPiece = Piece()
        self.pieces.append(newPiece)
        self.length += newPiece.l

    def findPoint(self, fraction):
        if not (0 <= fraction <= 1):
            raise ValueError('Fraction must be between 0 and 1')

        thisLength = fraction * self.length
        curLength = 0.
        coords = [0., 0.]
        for p in self.pieces:
            if thisLength <= curLength + p.l:
                coords += (thisLength - curLength) * p.v
                return coords
            else:
                curLength = curLength + p.l
                coords += p.v * p.l


class SpiralPiece():
    def __init__(self, length, vector, level):
        self.l = length
        self.v = np.array(vector)
        self.level = level
