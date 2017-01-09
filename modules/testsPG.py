__author__ = 'miguel.herraez'

def cases(case):

    particles = []

    if case == 1:
        # No intersection
        L, H = 10., 10.
        particles.append({'d':2.0, 'x0':5.0, 'y0':3.0})
    elif case == 2:
        # Simple intersection
        L, H = 10., 10.
        particles.append({'d':2.0, 'x0':5.2, 'y0':3.0})
        particles.append({'d':2.0, 'x0':6.0, 'y0':3.0})
    elif case == 3:
        # Simple periodicity
        L, H = 10., 10.
        particles.append({'d':2.0, 'x0':5.0, 'y0':0.5})
    elif case == 4:
        # Simple periodicity and intersection
        L, H = 10., 10.
        particles.append({'d':2.0, 'x0':5.0, 'y0':0.5})
        particles.append({'d':2.0, 'x0':5.0, 'y0':1.1})
    elif case == 5:
        # Simple periodicity and intersection losing periodicity
        L, H = 10., 10.
        particles.append({'d':2.0, 'x0':5.0, 'y0':0.5})
        particles.append({'d':2.0, 'x0':5.0, 'y0':1.1, 'fixed':True})
        particles.append({'d':2.0, 'x0':7.0, 'y0':9.1, 'fixed':True})
        particles.append({'d':2.0, 'x0':3.5, 'y0':-0.4, 'fixed':True})
    elif case == 6:
        # Intersection between two periodic particles (master-master)
        L, H = 10., 10.
        particles.append({'d':2.0, 'x0':5.0, 'y0':0.5})
        particles.append({'d':2.0, 'x0':4.6, 'y0':0.1})
    elif case == 7:
        # Intersection between two periodic particles (master-master)
        L, H = 10., 10.
        particles.append({'d':2.0, 'x0':5.0, 'y0':0.5})
        particles.append({'d':2.0, 'x0':4.6, 'y0':10.1})
    elif case == 12:
        # Simple intersection. LOBULAR
        L, H = 10., 10.
        particles.append({'d':2.0, 'x0':5.5, 'y0':3.2, 'phi':0., 'shape':'LOBULAR', 'parameters':[2,]})
        particles.append({'d':2.0, 'x0':6.0, 'y0':3.0, 'phi':0., 'shape':'LOBULAR', 'parameters':[2,]})
        particles.append({'d':2.0, 'x0':8.0, 'y0':2.8, 'phi':40, 'shape':'LOBULAR', 'parameters':[2,]})
        particles.append({'d':2.0, 'x0':5.0, 'y0':2.6, 'phi':90, 'shape':'LOBULAR', 'parameters':[2,]})
        particles.append({'d':2.0, 'x0':5.1, 'y0':2.6, 'phi':20, 'shape':'LOBULAR', 'parameters':[2,]})
    elif case == 13:
        # Simple intersection. LOBULAR
        L, H = 10., 10.
        particles.append({'d':2.0, 'x0':5.5, 'y0':3.2, 'phi':0., 'shape':'LOBULAR', 'parameters':[2,]})
        particles.append({'d':2.0, 'x0':6.0, 'y0':3.0, 'phi':0., 'shape':'LOBULAR', 'parameters':[2,]})
        particles.append({'d':2.0, 'x0':8.0, 'y0':2.8, 'phi':40, 'shape':'LOBULAR', 'parameters':[2,]})
        particles.append({'d':2.0, 'x0':5.0, 'y0':2.6, 'phi':90, 'shape':'LOBULAR', 'parameters':[2,]})
        particles.append({'d':2.0, 'x0':5.1, 'y0':2.6, 'phi':20, 'shape':'LOBULAR', 'parameters':[2,]})
        # particles.append({'d':2.0, 'x0':1.1, 'y0':2.4, 'phi':-20, 'shape':'LOBULAR', 'parameters':[2,]})
        particles.append({'d':2.0, 'x0':11.1, 'y0':2.4, 'phi':-20, 'shape':'LOBULAR', 'parameters':[2,]})
    else:
        raise IndexError('Case {} is not implemented'.format(case))

    return particles, (L,H)