# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:38:22 2015

@author: javierseguradoescudero
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from hard_sphere_mod import setPeriodic, updatePeriodics
# from myClasses import Dispersion

DEBUGGING = False
ITERATION = 0
LOG_FILE = 'PG.log'

def nonlinear_conjugate_gradient(dfun, fun, dispersion, tol, plot=False, verbose=True, maxiter=5000):
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    rm_old = -1.*dfun(dispersion)
    # if DEBUGGING:
    #     dispersion.plot(ion=True)
    rm = rm_old
    rm_scalar = np.linalg.norm(rm_old)
    pm_old = rm_old
    pm = pm_old
    V_old = fun(dispersion)
    V = V_old
    Vdif = 1.
    if verbose:
        print 'Initial residual = {0:6.3f}'.format(rm_scalar)
        print 'Initial potential = {0:6.3f}'.format(V_old)
    iter = 0

    logFile = open(LOG_FILE, 'w')

    if verbose:
        title = '{0:>5s} {1:>9s} {2:>15s} {3:>12s} {4:>17s} {5:>5s} {6:>6s} {7:>6s}'.format('iter', 'energy', 'residual', 'alpha',
                                                                            'np.norm(p)*alpha', 'time', 'loops1', 'loops2')
        print title,
        logFile.write(title)
        string = '\n{0:5d} {1:9.4f} {2:15.9f} {3:12.4f} {4:17.6f} {5:5.2f} {6:6d} {7:6d}'.format(iter, V_old, rm_scalar,
                        0., 0., 0., 0, 0)
        print string,
        logFile.write(string)

    # while res_scalar>tol and Vdif>tol:
    while rm_scalar>tol and Vdif>tol and iter<maxiter:

        t0 = time.time()
        iter += 1
        # Save initial position
        x_ini = np.array(dispersion.getCentres())

        pm_old = pm # Gradient (vector)
        rm_old, V_old = rm, V
        f0 = -rm
        alpha, loops1, loops2 = line_search_opt(dispersion, pm, V_old, f0, fun, dfun, .5, verbose=verbose)

        ### Update particles position
        dispersion.setCentres(x_ini, alpha*pm_old)

        # Update neighbours and periodicity
        # dispersion.update(periodic=True, neighbours=True)
        if (iter-1) % 4 == 0:
            dispersion.update(periodic=True, neighbours=True)
        #     dispersion.update(neighbours=True)

        rm = -1.*dfun(dispersion)
        V = fun(dispersion) # Energy of the system
        Vdif = abs((V-V_old)/V_old)
        rm_scalar = np.linalg.norm(rm)

        if verbose:
            string = '\n{0:5d} {1:9.4f} {2:15.9f} {3:12.4f} {4:17.6f} {5:5.2f} {6:6d} {7:6d}'.format(iter, V, rm_scalar,
                        alpha, np.linalg.norm(pm)*alpha, time.time()-t0, loops1, loops2)
            print string,
            logFile.write(string)

        # print '////////////////////////////////////////\n\n'
        # Update plot
        if (iter-1) % 4 == 0 and plot:
            dispersion.updatingPlot(ax)

        # Several options to choose beta:
        # beta = np.dot(rm,rm)/np.dot(rm_old,rm_old)
        beta = (np.dot(rm, rm)-np.dot(rm, rm_old)) / np.dot(rm_old, rm_old)
        pm = rm + beta*pm_old

        if DEBUGGING:
            dispersion.updateDisp(pm)
            dispersion.plot(ion=False, numbering=False, save='ITER_{0:03d}.png'.format(iter),
                            show=False, arrows=False, title=False, ticks=False)

        ### Finished condition
        if rm_scalar<tol or Vdif<tol:
            # Update and last check
            if (iter-1) % 4 != 0:
                dispersion.update(neighbours=True, periodic=True)
                rm = -1.*dfun(dispersion)
                V = fun(dispersion) # Energy of the system
                Vdif = abs((V-V_old)/V_old)
                rm_scalar = np.linalg.norm(rm)
                if rm_scalar<tol or Vdif<tol:
                    break
            else:
                break

    # Close log file
    logFile.close()

    # Last update
    dispersion.updateDisp(pm)
    dispersion.update(neighbours=True, periodic=True)

    return 0

def line_search_opt(dispersion, pm, V_old, f0, fun, dfun, rho, verbose=True):
    # Initial position
    x_ini = np.array(dispersion.getCentres())

    p_norm = np.linalg.norm(pm)
    f_dot_p = np.dot(f0, pm)
  
    # First find alpha1 and alpha1
    # Alphamin defined to have a minimum step equal to precision 1E-16
    alphamax = 1./p_norm
    alphamin = max(1.e-12, 1.e-9/p_norm)
    dalpha = max(1.e-10, 1.e-6/p_norm)
   
    # Increments in alpha to find parabola
    alpha = dalpha
    c1 = 1.0e-4
    phi0 = V_old
    phi1 = phi0 + .1*abs(phi0)
    iter1 = 0
    iter2 = 0
    alphap = 0

    ## print 'finding phi1'
    # print '-alpha in phi1', alpha, phi0, phi1, np.linalg.norm(alpha*pm)
    while phi1 > phi0:
        iter1 += 1
        dispersion.setCentres(x_ini, alpha*pm)
        phi1 = fun(dispersion)
        # print 'alpha in phi1', alpha, phi0, phi1, np.linalg.norm(alpha*pm)
        if iter1 < 2:
            alpha += dalpha
        else:
            alpha *= 1.6
        alpha1 = alpha
        if alpha*1.6 > alphamax:
            alpha = dalpha
            if verbose: print 'err LS1',
            dispersion.setCentres(x_ini, alpha*pm)
            funalpha = fun(dispersion)
            while funalpha > phi0 + c1*alpha*f_dot_p:
                 alpha = rho*alpha
                 # print 'alpha in backtrack', alpha, phi0, funalpha
                 dispersion.setCentres(x_ini, alpha*pm)
                 funalpha = fun(dispersion)
                 if alpha < alphamin:
                     alpha = alphamin
                     if verbose: print 'Error in backtrack',
                     break
            return alpha, iter1, iter2

    ## print 'finding phi2'
    phi2 = phi1-.1*abs(phi1)
    # print '-alpha in phi2', alpha, alphamax, phi0, phi1, phi2, iter2
    while phi2 < phi1:
        iter2 += 1
        alpha += dalpha
        if iter2 < 2:
            alpha += dalpha
        else:   
            alpha *= 1.6
        alpha2 = alpha
        dispersion.setCentres(x_ini, alpha*pm)
        phi2 = fun(dispersion)
        # print 'alpha in phi2', alpha, phi0, phi1, phi2, iter2
        if 1.6*alpha > alphamax:
            if verbose: print 'ERR LS2',alpha2,
            alphap = alpha2
#            return alphap
            break
 
    if alphap == 0:
        if phi0*(alpha2-alpha1) - phi1*alpha2 + phi2*alpha1 < 1.e-16:
            alphap = (alpha2+alpha1)/2.
        else:
            alphap = .5*(phi0*(alpha2**2-alpha1**2)-phi1*alpha2**2+phi2*alpha1**2)/(phi0*(alpha2-alpha1)-phi1*alpha2+phi2*alpha1)
  
    alpha = alphap
#    print 'phi0,phi1,phi2',phi0,phi1,phi2
#    print 'alphas',alpha1,alpha2,alphap,alphamin
    if alpha < alphamin:
        alpha = alphamin
    else:
        dispersion.setCentres(x_ini, alpha*pm)
        funalpha = fun(dispersion)
        f_dot_p = np.dot(f0, pm)
        while funalpha > phi0 + c1*alpha*f_dot_p:
            alpha = rho*alpha
#            print'alpha in backtrack',alpha,phi0,funalpha
            dispersion.setCentres(x_ini, alpha*pm)
            funalpha = fun(dispersion)
            if alpha < alphamin:
                alpha = alphamin
                if verbose: print 'Error in linesearch',
                break
#    print 'phi0,phi1,phi2',phi0,phi1,phi2
#    print 'alpha1,alpha2,alpha',alpha1,alpha2,alpha
    return alpha, iter1, iter2
