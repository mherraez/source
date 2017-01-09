__author__ = 'miguel.herraez'

# import matplotlib
# import matplotlib.backends.backend_tkagg
# matplotlib.use('TKAgg')
# matplotlib.use('Qt4Agg')
# matplotlib.rcParams['backend.qt4'] = 'PySide'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.patches import Polygon, Rectangle
# from matplotlib.widgets import Button
from shapely.geometry import Point
import numpy as np
# import copy
# Own built libraries
import RVE


# CLASSES-------------------------------------------

# Draggable polygons used to allocate fibers manually
class draggablePolygon():

    lock = None  # only one can be animated at a time
    copiedDp = None
    List = []
    microstructure = None
    title = "Drag (LB), Rotate (RB), Enlarge (MB), Delete('d') and Copy-Paste ('c'-'v')\n"

    def __init__(self, poly, ax, sq, i):

        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')

        draggablePolygon.List.append(self)
        self.poly = poly
        self.ax = ax
        self.sq = sq
        self.ind = i
        self.press = None
        self.connected = False

    def connect(self):
        'connect to all the events we need'
        self.connected = True
        self.cidpress = self.poly.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.poly.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.poly.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidkeys = self.poly.figure.canvas.mpl_connect('key_press_event', draggablePolygon.key_pressed)

    @staticmethod
    def attachMicrostructure(microstructure):
        draggablePolygon.microstructure = microstructure

    @staticmethod
    def key_pressed(event):
        """
        d: deletes fiber
        c: copy fiber
        v: paste copied fiber
        """
        if not event.inaxes: return

        if event.key == 'd':
            dp = draggablePolygon.get_dp_under_point(event)
            try:
                # if dp.connected:
                dp.destroy()
                print 'Delete fiber %d' % dp.ind
                # redraw the full figure
                dp.poly.figure.canvas.draw()
                # else:
                #     print 'Not editable fiber'
            except:
                print 'No fiber found'

        elif event.key == 'c':
            dp = draggablePolygon.get_dp_under_point(event)
            try:
                if dp.connected:
                    draggablePolygon.copiedDp = dp
                    print 'Copy fiber %d' % dp.ind
                else:
                    draggablePolygon.copiedDp = None
                    print 'Not editable fiber'
            except:
                print 'No fiber found'

        elif event.key == 'v':
            dp = draggablePolygon.get_dp_under_point(event)
            if not dp:
                if not draggablePolygon.copiedDp:
                    print "Copy a fiber first (press 'c')"
                else:
                    # newdp = draggablePolygon.copiedDp.deepcopy(event)
                    newfiber = RVE.Fiber(geometry=draggablePolygon.copiedDp.sq.geometry, parameters=draggablePolygon.copiedDp.sq.parameters, material=draggablePolygon.copiedDp.sq.material,
                             L=draggablePolygon.copiedDp.sq.L, phi=draggablePolygon.copiedDp.sq.phi, center=draggablePolygon.copiedDp.sq.center, period=draggablePolygon.copiedDp.sq.period, Nf=draggablePolygon.copiedDp.sq.Nf)
                    newfiber.set_center((event.xdata, event.ydata))
                    draggablePolygon.copiedDp.ax.add_patch(newfiber.poly)
                    newdp = draggablePolygon(newfiber.poly, draggablePolygon.copiedDp.ax, newfiber, i=len(draggablePolygon.List))
                    newdp.connect()

                    # Redraw polygon in the new position
                    canvas = newdp.poly.figure.canvas
                    axes = newdp.poly.axes
                    # redraw just the current polygon
                    newdp.poly.xy = np.transpose(newdp.sq.vertices)
                    axes.draw_artist(newdp.poly)
                    # blit just the redrawn area
                    canvas.blit(axes.bbox)

                    # Append copied fiber into microstructure
                    draggablePolygon.microstructure.sq.append(newfiber)
                    print 'Paste fiber'
            else:
                print 'Cannot paste fiber into another fiber'

        else:
            try:
                mat = int(event.key)
                dp = draggablePolygon.get_dp_under_point(event)
                if dp:
                    if mat in RVE.FIBERMATERIAL.keys():
                        print 'Fiber %d is %s' % (dp.ind, RVE.FIBERMATERIAL[mat])
                        material = RVE.FIBERMATERIAL[mat]
                        dp.sq.set_material(material)
                        # fiberColour, textColour, edgeColour = RVE.colourFiber(material)
                        # dp.poly.set_facecolor(fiberColour)
                        # print dp.poly.facecolor
                        # dp.poly.set_edgecolor(edgeColour)
                        # # redraw the full figure
                        # canvas = dp.poly.figure.canvas
                        # axes = dp.poly.axes
                        #
                        # axes.draw_artist(dp.poly)
                        # canvas.blit(axes.bbox)
                        dp.poly.figure.canvas.draw()
                    else:
                        print 'Material %d is not defined' % (mat)
                else:
                    print 'Point on a fibre to modify the material'
            except ValueError:
                pass

    @staticmethod
    def get_dp_under_point(event):

        point = Point(event.xdata, event.ydata)
        # print (event.xdata, event.ydata)
        dp = [dp for dp in draggablePolygon.List if dp.sq.polygonly.contains(point)]
        if not dp:
            dp = None
        else:
            dp = dp[0]
        return dp

    # @property
    # def deepcopy(self, event=None):
    #
    #     # newfiber = copy.deepcopy(self.sq)
    #
    #     newfiber = RVE.Fiber(geometry=self.sq.geometry, parameters=self.sq.parameters, material=self.sq.material,
    #                          L=self.sq.L, phi=self.sq.phi, center=self.sq.center, period=self.sq.period, Nf=self.sq.Nf)
    #     newfiber.set_center((event.xdata, event.ydata))
    #     polyvert = np.asarray(newfiber.polygonly.exterior)
    #     polyfiber = Polygon(polyvert, facecolor='w', edgecolor=None)
    #     self.ax.add_patch(polyfiber)
    #
    #     newdp = draggablePolygon(polyfiber, self.ax, newfiber, i=len(draggablePolygon.List))
    #     newdp.connect()
    #     return newdp

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.poly.axes: return

        contains, attrd = self.poly.contains(event)
        if not contains: return
        print '________________'
        print '- Selected fiber: %d, (%s)' % (self.ind, self.sq.material)
        print 'Center: (%5.3f, %5.3f)' % (self.sq.center[0],self.sq.center[1])
        x0, y0 = self.sq.center
        self.press = x0, y0, event.xdata, event.ydata

        # Animation to smooth fibre displacement
        draggablePolygon.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.poly.figure.canvas
        axes = self.poly.axes
        self.poly.set_animated(True)
        # Highlight moving fiber
        self.poly.set_edgecolor('green')
        self.poly.set_linewidth(3)
        self.poly.set_alpha(0.75)
        # Reprint title
        # self.ax.set_title(draggablePolygon.title + 'Selecting fiber %i' % self.ind)
        # Draw canvas
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.poly.axes.bbox)

        # now redraw just the polygon
        axes.draw_artist(self.poly)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.poly.axes: return
        x0, y0, xpress, ypress = self.press
        if event.button == 1:  # Drag fiber
            dx = event.xdata - xpress
            dy = event.ydata - ypress
    #        print 'x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f'%(x0, xpress, event.xdata, dx, x0+dx)
            self.sq.set_center((x0+dx, y0+dy))

        elif event.button == 2:  # Drag fiber horizontally
            dL = event.ydata - ypress
            # dy = event.ydata - ypress
    #        print 'x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f'%(x0, xpress, event.xdata, dx, x0+dx)
            self.sq.set_size(self.sq.L+0.2*dL)

        elif event.button == 3:  # Rotate fiber, phi = np.arctan2(y,x)
            apress = np.arctan2(ypress-y0, xpress-x0)
            adata = np.arctan2(event.ydata-y0, event.xdata-x0)
            da = adata - apress
            self.sq.set_phi(self.sq.phi+5*da)  # da is scaled to rotate faster

        # Redraw polygon in the new position
        canvas = self.poly.figure.canvas
        axes = self.poly.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current polygon
        self.poly.xy = np.transpose(self.sq.vertices)
        axes.draw_artist(self.poly)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if draggablePolygon.lock is not self:
            return
        self.press = None
        draggablePolygon.lock = None
        self.poly.set_edgecolor('black')
        self.poly.set_linewidth(1)
        self.poly.set_alpha(1)

        # turn off the rect animation property and reset the background
        self.poly.set_animated(False)
        self.background = None

        # redraw the full figure
        self.poly.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.connected = False
        self.poly.figure.canvas.mpl_disconnect(self.cidpress)
        self.poly.figure.canvas.mpl_disconnect(self.cidrelease)
        self.poly.figure.canvas.mpl_disconnect(self.cidmotion)

    def destroy(self):
        # Remove from list of draggable polygons
        draggablePolygon.List.remove(self)
        # Remove from fibers in microstructure
        draggablePolygon.microstructure.sq.remove(self.sq)
        # Remove polygon from figure
        self.poly.remove()

    @staticmethod
    def reset():
        draggablePolygon.List = []
        draggablePolygon.microstructure = None


class HighLight():

    def __init__(self):
        self.polygon = Polygon([[0, 0], [0, 0]], facecolor='lightyellow', edgecolor='green', linewidth=2)  # dummy data for xs,ys

    def update_polygon(self, tri):
        if tri == -1:
            points = [(0,0),(0,0)]
        else:
            points = draggablePolygon.List[tri].sq.vertices
            # dps[tri].poly.set_edgecolor('green')
            # dps[tri].poly.set_color('red')
        self.polygon.set_xy(zip(*points))
        self.polygon.set_zorder(tri+1e6)


class ManualEdition():

    def __init__(self, microstructure, highlight=False):

        """ Open new window in Matplotlib to edit manually """
        # Enable manual edition of the microstructure
        # plt.ion()
        self.title = "Drag (LB), Enlarge (MB), Rotate (RB),\n" \
                     "Delete('d'), Copy-Paste ('c'-'v'),\n" \
                     "Modify fiber material (1-5)"
        S0 = microstructure.rveSize   # Microstructure dimensions
        L = float(max(S0))
        lmax = 9.0
        sizex = S0[0]/L*lmax
        sizey = S0[1]/L*lmax

        # Plot configuration
        # self.fig, self.ax = plt.subplots(num=None, figsize=(sizex, sizey), dpi=80, facecolor='w', edgecolor='k')
        self.fig = plt.figure(num=None, figsize=(sizex, sizey), dpi=80, facecolor='w', edgecolor='k')
        self.ax = self.fig.add_axes([0.05, 0.1, 0.75, 0.8])
        self.ax.set_aspect('equal')
        self.ax.add_patch(Rectangle((0,0), S0[0], S0[1], facecolor='firebrick'))
        self.ax.set_title(self.title)
        self.ax.set_xlim((0,S0[0]))
        self.ax.set_ylim((0,S0[1]))
        self.ax.set_xticks([0,S0[0]])
        self.ax.set_yticks([0,S0[1]])
        self.legendHandler()

        # Window positioning
        # mngr = plt.get_current_fig_manager()
        # geom = mngr.window.geometry()
        # x, y, dx, dy = geom.getRect()
        # mngr.window.setGeometry(70,30,dx,dy)
        draggablePolygon.reset()

        # Fibers set up
        for i, fiber in enumerate(microstructure.sq):
            polyvert = np.asarray(fiber.polygonly.exterior)
            # polyfiber = fiber.poly
            fibercolour, textcolour, edgecolour = RVE.colourFiber(fiber.material)
            # polyfiber.set_facecolor(fibercolour)
            # polyfiber.set_edgecolor(edgecolour)

            if fiber.period == 0:
                polyfiber = Polygon(polyvert, facecolor=fibercolour, edgecolor=edgecolour)
                self.ax.add_patch(polyfiber)
                dp = draggablePolygon(polyfiber, self.ax, fiber, i)
                dp.connect()  # activation draggability
                # self.ax.text(fiber.center[0], fiber.center[1], 'O', color=textcolour, verticalalignment='center',
                #              horizontalalignment='center', weight='semibold')

            else:
                polyfiber = Polygon(polyvert, facecolor=fibercolour, edgecolor=fibercolour, lw=1.5)
                self.ax.add_patch(polyfiber)
                draggablePolygon(polyfiber, self.ax, fiber, i)
                self.ax.text(fiber.center[0], fiber.center[1], 'X', color=textcolour, verticalalignment='center',
                             horizontalalignment='center', weight='semibold')

        # Attach microstructure
        draggablePolygon.attachMicrostructure(microstructure)

        # Dummy polygon to highlight fibres
        if highlight:
            self.highlight = HighLight()
            self.ax.add_patch(self.highlight.polygon)
            self.highlight.update_polygon(-1)
            self.fig.canvas.mpl_connect('motion_notify_event', self.motion_notify)

        # Show microstructure
        plt.show()

    def legendHandler(self):
        legendPatches = list()
        for key, value in RVE.FIBERMATERIAL.items():
            fibrecolour, _, edgecolour = RVE.colourFiber(value)
            legendPatches.append(mpatches.Patch(facecolor=fibrecolour, label="%d - %s" % (key,value), edgecolor=edgecolour))
        leg = self.ax.legend(handles=legendPatches,
                             bbox_to_anchor=(1.01, 1.),
                             loc='upper left', borderaxespad=0.,
                             prop={'size':'small'})
        # leg = self.ax.legend(handles=legendPatches)
        leg.draggable()

    def motion_notify(self, event):
        tri = -1
        if event.inaxes:
            point = Point(event.xdata, event.ydata)

            for i, dp in enumerate(draggablePolygon.List):
                if point.within(dp.sq.polygonly) and dp.connected:
                    tri = i
                    self.ax.set_title(self.title+'Selecting fiber %i' % tri)
                    break
                else:
                    self.ax.set_title(self.title)
        self.highlight.update_polygon(tri)
        event.canvas.draw()


#===============================================================================
if __name__ == "__main__":
    import os
    # plt.close('all')
    filename = r'lob4'
    diretory = r'C:\Users\miguel.herraez\Desktop\VIPPER_project - shapely\Microstructures'
    myMicro = RVE.Microstructure(read_microstructure = os.path.join(diretory, filename))
    ManualEdition(myMicro)

#--EOF---