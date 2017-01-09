__appName__ = "ViP3r"
__appVersion__ = "0.2"
__module__ = "main"
__author__ = 'Miguel.Herraez'

#----- START OF IMPORTS

import matplotlib
# import matplotlib.backends.backend_tkagg
# matplotlib.use('TKAgg')
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Polygon as mplPolygon
from matplotlib.patches import Circle as mplCircle
from matplotlib.patches import Rectangle
# from shapely.geometry import Point

from PySide.QtCore import *
from PySide.QtGui import *

import sys
import os
from my_uis import mainWindow, newRveWindow, addParamLobular, addParamElli, addParamOval, addParamSpoly, addParamPoly, addParamCshape
from my_uis import plotSettings, patternSettings, compactSettings, localVolumeFraction
from modules import RVE, interactiveEdit, texting
import numpy as np
import webbrowser
# import time

#----- END OF IMPORTS


#----- CONSTANTS
lmax = 6.0  # inches

# from matplotlib.backends import qt4_compat
# use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
# print use_pyside
# if use_pyside:  # Importing QtGui and QtCore from PySide or PyQt4
#     from PySide.QtCore import *
#     from PySide.QtGui import *
# else:
#     from PyQt4.QtCore import *
#     from PyQt4.QtGui import *


class Main(QMainWindow, mainWindow.Ui_MainWindow):
    """ Main window class of the application """

    plotSettings = {'COLOUR': True, 'NUMBERED': False, 'TICKS': True,
                    'COM': False, 'MOI': False}

    analysisStats = {'LVF_RADIUS': 10., 'LVF_SPACING': 1.}

    # rveSize = (0.0, 0.0)
    # rve_specifications = None

    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle(__appName__)
        self.myMicrostructure = None  #RVE.Microstructure()
        self.rve_specifications = rveSpecification()
        # self.rve_specifications.algorithm = {'GENERATION': 0, 'COMPACTION': 0}
        # self.rve_specifications.configuration = {'FiberSets': [], 'Tolerance': 0.1, 'Size': (58.0, 58.0)}
        # self.rveSize = (0.0, 0.0)

        # Matplotlib widget for main window
        layout = QVBoxLayout()
        self.sc = MyMplCanvas(self.widget, width=5, height=5, dpi=100)
        self.toolbar = NavigationToolbar(self.sc, self)
        layout.addWidget(self.sc)
        layout.addWidget(self.toolbar)
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)

        # Connections
        self.actionExit.triggered.connect(self.exit_action_triggered)
        self.actionNew.triggered.connect(self.new_microstructure)
        self.actionSave.triggered.connect(self.save_microstructure)
        self.actionOpen.triggered.connect(self.open_microstructure)
        self.actionPlotSettings.triggered.connect(self.viewSettings)
        self.actionAdd_fibres.triggered.connect(self.addFibers)
        self.actionCompact.triggered.connect(self.compactFibers)
        self.actionHelp.triggered.connect(self.getDocumentation)
        self.actionPattern.triggered.connect(self.patternRVE)
        self.actionEditManual.triggered.connect(self.bridge_editManual)
        self.actionRefresh.triggered.connect(self.refresh)
        self.actionSaveSimscreen.triggered.connect(self.saveSimscreen_microstructure)
        self.actionOpenSimscreen.triggered.connect(self.openSimscreen_microstructure)
        self.actionDiameters.triggered.connect(self.showDiameters)
        self.actionNNA.triggered.connect(self.showNNA)
        self.actionRDF.triggered.connect(self.showRDF)
        self.actionLocal_volume_fraction.triggered.connect(self.showLocalVf)

    def showLocalVf(self):
        if not self.myMicrostructure:
            print 'Microstructure unavailable'
            return

        # Input spot radius and spot spacing
        if AnalyseLocalVf(self.myMicrostructure.rveSize).exec_():
            spotSize = Main.analysisStats['LVF_RADIUS']
            spotSpacing = Main.analysisStats['LVF_SPACING']

            # Compute field
            x, y, myLvf = self.myMicrostructure.localVolumeFraction(spotSize=spotSize, voxelSize=spotSpacing,
                                                                    spots=True, fibres=True)
            print 'Bounds: {0:.1f}% < Vf < {1:.1f}%'.format(np.min(myLvf)*100, np.max(myLvf)*100)

            # TODO write results in text file
        else:
            print 'Cancelled'

    def showNNA(self):
        if not self.myMicrostructure:
            print 'Microstructure unavailable'
            return

        # Compute Nearest Neighbours distribution
        dn1 = self.myMicrostructure.analyzeNearestNeighbour(neighbour=1, show_plot=False)
        dn2 = self.myMicrostructure.analyzeNearestNeighbour(neighbour=2, show_plot=False)

        # Plot results
        fig, ax = plt.subplots()
        n = 25

        mu1 = dn1.mean()
        sigma1 = dn1.std()
        count, bins, ignored = ax.hist(dn1, n, color='blue', normed=True, alpha=0.5)
        ax.plot(bins, 1/(sigma1 * np.sqrt(2 * np.pi)) *
                 np.exp( - (bins - mu1)**2 / (2 * sigma1**2) ),
                 linewidth=2, color='r', label="1st")

        mu2 = dn2.mean()
        sigma2 = dn2.std()
        count, bins, ignored = ax.hist(dn2, n, color='grey', normed=True, alpha=0.5)
        ax.plot(bins, 1/(sigma2 * np.sqrt(2 * np.pi)) *
                 np.exp( - (bins - mu2)**2 / (2 * sigma2**2) ),
                 linewidth=2, color='k', label="2nd")

        ax.set_title("Nearest neighbour")
        ax.legend()
        plt.show()

    def showDiameters(self):
        if not self.myMicrostructure:
            return

        self.myMicrostructure.analyzeDiameters(show_plot=True)

    def showRDF(self):
        if not self.myMicrostructure:
            print 'Microstructure unavailable'
            return

        # Compute Radial distribution function (G) and second-order intensity function (K)
        rmax = 0.9*max(self.myMicrostructure.rveSize)
        dr = 0.1
        self.myMicrostructure.radialDistFunction(rmax, dr, plot=True, verbose=False, save='')

    def refresh(self):
        if self.myMicrostructure:
            self.plot_rve()

    def bridge_editManual(self):
        interactiveEdit.ManualEdition(self.myMicrostructure, True)

    @staticmethod
    def getDocumentation():
        """
        Opens the web browser with html-format documentation
        """
        url = r'documentation\home.html'
        webbrowser.open(url)

    def patternRVE(self):
        """
        Generates patterned RVE
        """
        pattern = PatternSettings()
        if pattern.exec_():
            self.myMicrostructure = RVE.patternedMicrostructure(self.myMicrostructure, ncols=pattern.settings['ncols'],
                                                                nrows=pattern.settings['nrows'], save=False, plot=False)
            self.plot_rve()
        else:
            print 'No pattern was applied'

    def compactFibers(self):
        compactSpec = CompactSettings(self.myMicrostructure)
        if compactSpec.exec_():

            print '\n___COMPACTING___'
            print 'Algorithm: %s\n' % (compactSpec.algorithm['COMPACTION'][0])

            if compactSpec.algorithm['COMPACTION'][0] == RVE.POINT:
                # TODO modify displacement protocol of periodic fibres
                self.myMicrostructure.compact_RVE(point=compactSpec.algorithm['COMPACTION'][1])

            elif compactSpec.algorithm['COMPACTION'][0] == RVE.DIRECTION:
                self.myMicrostructure.compact_RVE(vector=compactSpec.algorithm['COMPACTION'][1])

            elif compactSpec.algorithm['COMPACTION'][0] == RVE.STIRRING:
                # stirring algorithm
                self.myMicrostructure.stirringAlgorithm(eff=compactSpec.algorithm['COMPACTION'][1]/100.0)

            elif compactSpec.algorithm['COMPACTION'][0] == RVE.BOX2D:
                # Box2D algorithm
                gravity = compactSpec.algorithm['COMPACTION'][1]
                shake = compactSpec.algorithm['COMPACTION'][2]
                # V_Box2D.compactBox2D(self.myMicrostructure, gravity=gravity, shake=shake)
                RVE.compactBox2D(self.myMicrostructure, gravity=gravity, shake=shake)

            else:
                print 'No selection'
                return
            self.plot_rve()

    def addFibers(self):
        newSpecifications = AddFibers(self.myMicrostructure)
        if newSpecifications.exec_():
            self.join_fiberSets(newSpecifications)
            self.myMicrostructure.add_fiber_sets(newSpecifications.configuration['FiberSets'], newSpecifications.algorithm['GENERATION'],
                                                 newSpecifications.algorithm['COMPACTION'], newSpecifications.algorithm['PERIODICITY'])
            self.plot_rve()
        # else:
        #     print 'No fiber was added'

    def join_fiberSets(self, newSpecifications):
        self.rve_specifications.algorithm['GENERATION'] = newSpecifications.algorithm['GENERATION']
        self.rve_specifications.algorithm['COMPACTION'] = newSpecifications.algorithm['COMPACTION']

        self.rve_specifications.configuration['FiberSets'].append(newSpecifications.configuration['FiberSets'])

    def viewSettings(self):
        """

        """
        PlotSettings().exec_()
        if self.myMicrostructure:
            self.plot_rve()         # Update microstructure view with new settings

    def new_microstructure(self):
        self.rve_specifications = NewMicrostructure()
        if self.rve_specifications.exec_():
            print 'Generating microstructure'
            # Get new microstructure and plot it in QWidget
            # self.label1.setText('Spinbox value is: ' + str(newMicrostructure.spinbox.value()))
            print self.rve_specifications.algorithm         # Generation (and compaction) algorithms
            print self.rve_specifications.configuration     # Fiber sets to generate and RVE size
            self.myMicrostructure = RVE.Microstructure(rve_size=self.rve_specifications.configuration['Size'],
                                                       fibersets=self.rve_specifications.configuration['FiberSets'],
                                                       tolerance=self.rve_specifications.configuration['Tolerance'],
                                                       gen_algorithm=self.rve_specifications.algorithm['GENERATION'],
                                                       comp_algorithm=self.rve_specifications.algorithm['COMPACTION'],
                                                       optPeriod=self.rve_specifications.algorithm['PERIODICITY'])
            # self.myMicrostructure.save_rve(filename='Testing microstructure')

            self.plot_rve()
        else:
            # QMessageBox.warning(self, __appName__, 'Dialog cancelled')
            # print 'Dialog was cancelled'
            pass

    def save_microstructure(self, directory=r'.\Microstructures'):
        """ Saves current Microstructure in text file """
        # Do not save when 'Cancel' (name == '') is pressed
        if bool(self.myMicrostructure):
            filePath = QFileDialog.getSaveFileName(self, __appName__ + ' Open file dialog', dir=directory,
                                                   filter='Text files (*.txt)')
            #print 'File path:', filePath
            if filePath[0]:
                directory, filename = os.path.split(filePath[0])
                shortName, extension = os.path.splitext(filename)
                # print filename
                self.myMicrostructure.save_rve(filename=shortName, directory=directory)
                # self.myMicrostructure.plot_rve(filename=shortName, directory=directory, imageFormat='pdf', save=True,
                #                                show_plot=False, numbering=False)

    def saveSimscreen_microstructure(self, directory=r'.\Microstructures'):
        """ Saves current Microstructure in text file """
        # Do not save when 'Cancel' (name == '') is pressed
        if bool(self.myMicrostructure):
            filePath = QFileDialog.getSaveFileName(self, __appName__ + ' Open file dialog', dir=directory,
                                                   filter='Text files (*.txt)')
            #print 'File path:', filePath
            if filePath[0]:
                directory, filename = os.path.split(filePath[0])
                shortName, extension = os.path.splitext(filename)
                # print filename
                self.myMicrostructure.saveSimscreen_rve(filename=shortName, directory=directory)
                self.myMicrostructure.plot_rve(filename=shortName, directory=directory, imageFormat='pdf', save=True,
                                               show_plot=False, numbering=False)

    def open_microstructure(self, directory=r'.\Microstructures'):
        """ Saves current Microstructure in text file """
        if bool(self.myMicrostructure):
            pass
            # Delete current microstructure
        filePath = QFileDialog.getOpenFileName(self, __appName__ + ' Open file dialog', dir=directory,
                                              filter='Text files (*.txt)')
        directory, filename = os.path.split(filePath[0])
        if filename:
            shortName, extension = os.path.splitext(filename)
            self.myMicrostructure = RVE.Microstructure(read_microstructure=os.path.join(directory, shortName))
            # Read specifications from txt file and redefine rveSize
            self.rveSize = self.myMicrostructure.rveSize
            self.rve_specifications.configuration['Size'] = self.rveSize
            print self.rveSize
            self.plot_rve()

    def openSimscreen_microstructure(self, directory=r'.\Microstructures'):
        """ Saves current Microstructure in text file """
        # if bool(self.myMicrostructure):
        #     pass
            # Delete current microstructure
        filePath = QFileDialog.getOpenFileName(self, __appName__ + ' Open file dialog', dir=directory,
                                              filter='Text files (*.txt)')
        directory, filename = os.path.split(filePath[0])
        shortName, extension = os.path.splitext(filename)
        self.myMicrostructure = RVE.Microstructure()
        self.myMicrostructure.readSimscreen_rve(directory, shortName)
        # self.myMicrostructure = RVE.Microstructure(read_microstructure=os.path.join(directory, shortName))
        # Read specifications from txt file and redefine rveSize
        self.rveSize = self.myMicrostructure.rveSize
        print self.rveSize
        self.plot_rve()

    def plot_rve(self):
        self.sc.update_figure(microstructure=self.myMicrostructure)
        # self.sc.update_figure(rveSize=size, fibers=self.myMicrostructure.sq)
        # Resize mainwindow to fit current RVE size
        # L = float(max(self.myMicrostructure.rveSize))
        # sizex = self.myMicrostructure.rveSize[0]/L*lmax * 100  # pixels
        # sizey = self.myMicrostructure.rveSize[1]/L*lmax * 100  # pixels
        w, h = self.size().toTuple()
        # print w, h
        self.resize(w-1, h-1)
        self.resize(w, h)
        self.actionAdd_fibres.setEnabled(True)
        self.actionCompact.setEnabled(True)
        self.actionPattern.setEnabled(True)
        self.actionEditManual.setEnabled(True)
        # self.adjustSize()
        # self.repaint()

    def exit_action_triggered(self):
        """Closes the application"""
        self.close()

    def closeEvent(self, event, *args, **kwargs):
        """Overrides the default close method"""

        result = QMessageBox.question(self, __appName__, "Are you sure you want to exit?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        if result == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


class AnalyseLocalVf(QDialog, localVolumeFraction.Ui_localVf):
    """ Dialog for compaction """

    def __init__(self, size, parent=None):
        super(AnalyseLocalVf, self).__init__(parent)
        self.setupUi(self)

        self.size = size

        # Set spinbox values
        self.spinBox_spotSize.setProperty("value", Main.analysisStats['LVF_RADIUS'])
        self.spinBox_spotSpacing.setProperty("value", Main.analysisStats['LVF_SPACING'])

        self.pushButton_ok.clicked.connect(self.accept)


    def accept(self, *args, **kwargs):

        class GridSpacing(Exception): pass

        # Check arguments are compatible and make sense
        try:
            if (self.spinBox_spotSpacing.value() >= min(self.size)):
                raise GridSpacing, "Grid spacing must be smaller than width and height of the RVE"
            if (self.spinBox_spotSpacing.value() <= 0.) or (self.spinBox_spotSize <= 0.):
                raise ZeroDivisionError, "Spot radius and spacing must be above zero"
            else:
                Main.analysisStats['LVF_RADIUS'] = self.spinBox_spotSize.value()
                Main.analysisStats['LVF_SPACING'] = self.spinBox_spotSpacing.value()
                QDialog.accept(self)  # accept

        except GridSpacing, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.spinBox_spotSpacing.selectAll()
            self.spinBox_spotSpacing.setFocus()

        except ZeroDivisionError, e:
            QMessageBox.warning(self, __appName__, str(e))


class CompactSettings(QDialog, compactSettings.Ui_Dialog):
    """ Dialog for compaction """

    def __init__(self, myMicro, parent=None):
        super(CompactSettings, self).__init__(parent)
        self.setupUi(self)

        self.algorithm = {'GENERATION': [], 'COMPACTION': [], 'PERIODICITY': RVE.FULL}
        self.myMicro = myMicro

        if not all([fiber.geometry.upper() == RVE.CIRCULAR for fiber in self.myMicro.sq]):
            self.radioButton_compBox2D.setEnabled(False)

        # Connections
        self.radioButton_compPoint.clicked.connect(self.compactionChanged)
        self.radioButton_compDirection.clicked.connect(self.compactionChanged)
        self.radioButton_compStir.clicked.connect(self.compactionChanged)
        self.radioButton_compBox2D.clicked.connect(self.compactionChanged)

    def compactionChanged(self):
        """ Enables options for compaction methods """

        # Compaction - Point
        activePoint = self.radioButton_compPoint.isChecked()
        self.label_compPoint.setEnabled(activePoint)
        self.lineEdit_pointx.setEnabled(activePoint)
        self.lineEdit_pointy.setEnabled(activePoint)
        # if activePoint:
        #     print 'Point compaction'

        # Compaction - Direction
        activeDirection = self.radioButton_compDirection.isChecked()
        self.label_compDirection.setEnabled(activeDirection)
        self.lineEdit_vectorx.setEnabled(activeDirection)
        self.lineEdit_vectory.setEnabled(activeDirection)
        # if activeDirection:
        #     print 'Direction compaction'

        # Compaction - Stir
        activeStir = self.radioButton_compStir.isChecked()
        self.label_compStir.setEnabled(activeStir)
        self.doubleSpinBox_compStir.setEnabled(activeStir)
        # if activeDirection:
        #     print 'Stirring compaction'

        # Compaction - Box2D
        activeBox2D = self.radioButton_compBox2D.isChecked()
        self.label_gravity.setEnabled(activeBox2D)
        self.lineEdit_gravx.setEnabled(activeBox2D)
        self.lineEdit_gravy.setEnabled(activeBox2D)
        self.checkBox_shake.setEnabled(activeBox2D)
        # if activeBos2D:
        #     print 'Physical compaction'

    def accept(self, *args, **kwargs):

        # Compaction algorithm
        if self.radioButton_compPoint.isChecked():
            coord_x = float(self.lineEdit_pointx.text())
            coord_y = float(self.lineEdit_pointy.text())
            self.algorithm['COMPACTION'] = (RVE.POINT, (coord_x, coord_y))
        elif self.radioButton_compDirection.isChecked():
            coord_x = float(self.lineEdit_vectorx.text())
            coord_y = float(self.lineEdit_vectory.text())
            self.algorithm['COMPACTION'] = (RVE.DIRECTION, (coord_x, coord_y))
        elif self.radioButton_compStir.isChecked():
            effect = float(self.doubleSpinBox_compStir.value())/100.0
            self.algorithm['COMPACTION'] = (RVE.STIRRING, effect)
        elif self.radioButton_compBox2D.isChecked():
            grav_x = float(self.lineEdit_gravx.text())
            grav_y = float(self.lineEdit_gravy.text())
            shake = self.checkBox_shake.isChecked()
            self.algorithm['COMPACTION'] = (RVE.BOX2D, (grav_x, grav_y), shake)
        else:
            self.algorithm['COMPACTION'] = None

        # Check compaction arguments are compatible and make sense
        try:
            if (self.algorithm['COMPACTION']==(RVE.DIRECTION,(0,0))):
                raise ZeroVector, "Directional compaction algorithm requires not null vector"
            elif (self.algorithm['COMPACTION'][0] == RVE.BOX2D) and not all([fiber.geometry.upper() == RVE.CIRCULAR for fiber in self.myMicro.sq]):
                raise OnlyCircular, "Gravitational compaction is only supported for circular fibers"
            elif self.algorithm['COMPACTION'] == (RVE.BOX2D, (0, 0), False):
                raise NullCompaction, "No compaction will be performed"
            else:
                QDialog.accept(self)  # accept

        except ZeroVector, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.lineEdit_vectorx.selectAll()
            self.lineEdit_vectorx.setFocus()

        except OnlyCircular, e:
            QMessageBox.warning(self, __appName__, str(e))

        except NullCompaction, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.radioButton_compBox2D.setFocus()


class PatternSettings(QDialog, patternSettings.Ui_Dialog):
    """ Pattern settings dialog.
       - Number of rows
       - Number of columns
    """

    def __init__(self, parent=None):
        super(PatternSettings, self).__init__(parent)
        self.setupUi(self)

        self.settings = {'ncols': 1, 'nrows': 1}

    def accept(self, *args, **kwargs):
        self.settings['ncols'] = self.spinBoxCols.value()
        self.settings['nrows'] = self.spinBoxRows.value()
        QDialog.accept(self)


class rveSpecification(object):

    def __init__(self):
        self.configuration = {'Size': (0.0, 0.0), 'Tolerance': 0.0, 'FiberSets': []}
        self.algorithm = {'GENERATION': [], 'COMPACTION': [], 'PERIODICITY': RVE.FULL}


class PlotSettings(QDialog, plotSettings.Ui_plotSettings):
    """ Plot settings dialog. Available options are:
         - Colour
     - Numbering of fibres
     - Show ticks, labels and title
    """

    def __init__(self, parent=None):
        super(PlotSettings, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('View settings')

        # Plot settings
        self.checkBox_colour.setChecked(Main.plotSettings['COLOUR'])
        self.checkBox_numbered.setChecked(Main.plotSettings['NUMBERED'])
        self.checkBox_ticks.setChecked(Main.plotSettings['TICKS'])
        self.checkBox_COM.setChecked(Main.plotSettings['COM'])
        self.checkBox_MOI.setChecked(Main.plotSettings['MOI'])

    def accept(self, *args, **kwargs):
        Main.plotSettings['COLOUR'] = self.checkBox_colour.isChecked()
        Main.plotSettings['NUMBERED'] = self.checkBox_numbered.isChecked()
        Main.plotSettings['TICKS'] = self.checkBox_ticks.isChecked()
        Main.plotSettings['COM'] = self.checkBox_COM.isChecked()
        Main.plotSettings['MOI'] = self.checkBox_MOI.isChecked()
        QDialog.accept(self)  # accept method from QDialog!


class MyMplCanvas(FigureCanvas):
    """ Canvas with the RVE """

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='w', edgecolor='k')
        self.axes = self.fig.add_subplot(111)
        imgdirectory = r'.\my_uis\icons\viper.png'
        try:
            img = matplotlib.image.imread(imgdirectory)
            self.axes.imshow(img)
        except:
            print 'Warning: Cannot find %s' % imgdirectory
        self.axes.axis('off')
        # We want the axes cleared every time plot() is called
        # self.axes.hold(False)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        # self.compute_initial_figure()
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_figure(self, microstructure):
        self.axes.clear()
        L = float(max(microstructure.rveSize))
        w, h = microstructure.rveSize

        sizex = w/L*lmax
        sizey = h/L*lmax
        self.axes.set_xlim((0, w))
        self.axes.set_ylim((0, h))
        self.axes.set_aspect('equal')
        self.fig.set_size_inches(sizex, sizey)

        matrixColour = Main.plotSettings['COLOUR'] and 'dimgrey' or 'white'

        self.axes.add_patch(Rectangle((0, 0), w, h, facecolor=matrixColour))

        # Fibers are drawn
        for i, fiber in enumerate(microstructure.sq):

            if not Main.plotSettings['COLOUR']:
                fibercolour = 'white'
                textcolour = 'black'
                edgecolour = 'black'
            else:
                fibercolour, textcolour, edgecolour = RVE.colourFiber(fiber.material)

            # fiberpoly  = fiber.poly
            # fiberpoly.set_facecolor(fibercolour)
            if fiber.geometry.upper() == RVE.CIRCULAR:
                fiberpoly = mplCircle(fiber.center, radius=0.5*fiber.L, facecolor=fibercolour, edgecolor=edgecolour)
            else:
                polyvert = np.asarray(fiber.polygonly.exterior)
                fiberpoly = mplPolygon(polyvert, facecolor=fibercolour, edgecolor=edgecolour)
            self.axes.add_patch(fiberpoly)

            if Main.plotSettings['NUMBERED']:
                self.axes.text(fiber.center[0], fiber.center[1], i+1, color=textcolour,
                               verticalalignment='center', horizontalalignment='center', weight='semibold')

        if Main.plotSettings['MOI'] or Main.plotSettings['COM']:
            x0, y0 = microstructure.analyzeCenterOfMass()
            if Main.plotSettings['COM']:
                self.axes.plot(x0, y0, 'rx', mew=4, ms=10)
                self.axes.text(1.01*x0, 1.01*y0, '(%.2f, %.2f)' % (x0,y0), color='black', verticalalignment='bottom', fontsize=16,
                                bbox=dict(facecolor='w', alpha=0.2, ec='y'), horizontalalignment='left')

            if Main.plotSettings['MOI']:
                self.axes.axhline(y=y0, color='r', ls='--')
                self.axes.axvline(x=x0, color='r', ls='-.')
                Ix, Iy = microstructure.analyzeMomentsOfInertia()
                self.axes.text(1.01*w, y0, '$I_x$ = %.0f' % Ix, color='black', verticalalignment='center', fontsize=16,
                                bbox=dict(facecolor='w', alpha=0.5, ec='w'), horizontalalignment='left')
                self.axes.text(x0, -0.01*h, '$I_y$ = %.0f' % Iy, color='black', verticalalignment='top', fontsize=16,
                                bbox=dict(facecolor='w', alpha=0.5, ec='w'), horizontalalignment='center')

        # Compute fiber volume fraction
        vf, n = microstructure.fiber_volume()

        if Main.plotSettings['TICKS']:
            self.axes.set_xticks([0, w])
            self.axes.set_yticks([0, h])
            self.axes.set_title('$V_f = {0:.1f}\%$\t$N = {1:d}$'.format(vf*100, n))
        else:
            self.axes.set_xticks([])
            self.axes.set_yticks([])
            self.axes.set_title('')
            # self.axes.axis('off')

        self.draw()


class NewMicrostructure(QDialog, newRveWindow.Ui_Dialog, rveSpecification):
    """ New microstructure window """

    def __init__(self, parent=None):
        super(NewMicrostructure, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('New Microstructure Dialog')

        # Fiber description
        self.parameters = []
        self.geometry = self.comboBox_geometry.currentText()
        self.vf = self.spinBox_Vf.value()
        self.df = self.spinBox_df.value()
        self.D_df = self.spinBox_D_df.value()
        self.D_phi = self.spinBox_D_phi.value()
        self.phi = self.spinBox_phi.value()
        self.material = self.comboBox_material.currentText()
        self.sizeRve = [self.spinBox_SizeX, self.spinBox_SizeY]
        self.configuration = {'Size': self.sizeRve, 'Tolerance': self.spinBox_tolerance.value(), 'FiberSets': []}
        self.algorithm = {'GENERATION': (), 'COMPACTION': (), 'PERIODICITY': RVE.FULL}

        # Connections
        self.comboBox_geometry.currentIndexChanged.connect(self.geometry_changed)
        self.pushButton_add.clicked.connect(self.add_row)
        self.pushButton_remove.clicked.connect(self.remove_row)
        # Monitor compaction
        self.checkBox_compaction.stateChanged.connect(self.compactionChanged)
        # Monitor compaction algorithm
        self.radioButton_compPoint.clicked.connect(self.compactionChanged)
        self.radioButton_compDirection.clicked.connect(self.compactionChanged)
        self.radioButton_compStir.clicked.connect(self.compactionChanged)
        self.radioButton_compBox2D.clicked.connect(self.compactionChanged)
        # Monitor generation
        self.radioButton_randomGen.clicked.connect(self.generationChanged)
        self.radioButton_NNAgen.clicked.connect(self.generationChanged)
        self.radioButton_periodicSquareGen.clicked.connect(self.generationChanged)
        self.radioButton_periodicHexagonalGen.clicked.connect(self.generationChanged)
        self.radioButton_PotGen.clicked.connect(self.generationChanged)
        self.radioButton_DynGen.clicked.connect(self.generationChanged)


    def geometry_changed(self):
        self.parameters = []
        self.geometry = self.comboBox_geometry.currentText()
        try:  # Disconnect previous signal if exists
            self.pushButton_addParams.clicked.disconnect(self.connect_add_params)
        except RuntimeError:
            pass
        if self.geometry not in 'Circular':
            self.pushButton_addParams.setEnabled(True)
            self.pushButton_addParams.clicked.connect(self.connect_add_params)
        else:
            self.pushButton_addParams.setEnabled(False)

    def connect_add_params(self):
        if self.geometry in 'Lobular':
            addparams = LobularAddParams()
        elif self.geometry in 'Polygonal':
            addparams = PolygonAddParams()
        elif self.geometry in 'Spolygonal':
            addparams = SpolygonAddParams()
        elif self.geometry in 'Elliptical':
            addparams = EllipticalAddParams()
        elif self.geometry in 'Oval':
            addparams = OvalAddParams()
        elif self.geometry in 'CShape':
            addparams = CshapeAddParams()
        else:
            raise ValueError, '%s is not available' % self.geometry

        if addparams.exec_():
            self.parameters = addparams.parameters

        print self.geometry
        print self.parameters

    def add_row(self):
        """ Add the current fiber configuration to the table to generate the microstructure """
        if not self.check_input:
            return False

        # print 'Input is consistent'
        currentRowCount = self.tableWidget.rowCount()
        parameters = str(self.parameters) if self.parameters else None
        self.tableWidget.insertRow(currentRowCount)
        self.tableWidget.setItem(currentRowCount, 0, QTableWidgetItem(self.geometry))
        self.tableWidget.setItem(currentRowCount, 1, QTableWidgetItem(parameters))
        self.tableWidget.setItem(currentRowCount, 2, QTableWidgetItem(str(self.df)))
        self.tableWidget.setItem(currentRowCount, 3, QTableWidgetItem(str(self.D_df)))
        self.tableWidget.setItem(currentRowCount, 4, QTableWidgetItem(str(self.vf)))
        self.tableWidget.setItem(currentRowCount, 5, QTableWidgetItem(str(self.phi)))
        self.tableWidget.setItem(currentRowCount, 6, QTableWidgetItem(str(self.D_phi)))
        self.tableWidget.setItem(currentRowCount, 7, QTableWidgetItem(self.material))

        # Append current fiber configuration
        currentFiberSet = self.fiber_set()
        self.configuration['FiberSets'].append(currentFiberSet)
        # print currentFiberSet
        # print self.tableWidget.item(0, 0).text()  # Geometry

    def fiber_set(self):
        """ Regroup current fiber set parameters into a dictionary """
        return dict(Geometry=self.geometry,
                    Parameters=self.parameters,
                    df=(self.df, self.D_df),
                    Vf=self.vf,
                    Phi=(self.phi, self.D_phi),
                    Material=self.material)

    @property
    def check_input(self):
        """ Check input configuration validity """

        self.geometry = self.comboBox_geometry.currentText()
        self.df = self.spinBox_df.value()
        self.D_df = self.spinBox_D_df.value()
        self.vf = self.spinBox_Vf.value()
        self.phi = self.spinBox_phi.value()
        self.D_phi = self.spinBox_D_phi.value()
        self.material = self.comboBox_material.currentText()

        # Equivalent diameter cannot be equal to zero
        class ZeroSize(Exception): pass

        # Check if additional parameters are needed (depends on fiber geometry)
        class MissingParameters(Exception): pass

        # Fiber volume fraction cannot be equal to zero
        class ZeroVolumeFraction(Exception): pass

        try:
            if self.df == 0:
                raise ZeroSize, 'Equivalent diameter (df) cannot be equal to 0'
            if self.geometry not in 'Circular':
                if not self.parameters:
                    raise MissingParameters, 'Additional parameters are required for the selected geometry'
                else:
                    return True
            if self.vf == 0:
                raise ZeroVolumeFraction, 'Fiber volume fraction (Vf) cannot be equal to 0'

        except ZeroSize, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.spinBox_df.selectAll()
            self.spinBox_df.setFocus()
            return False

        except MissingParameters, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.pushButton_addParams.setFocus()
            return False

        except ZeroVolumeFraction, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.spinBox_Vf.selectAll()
            self.spinBox_Vf.setFocus()
            return False

        return True

    def remove_row(self):
        """Removes the selected row from the mainTable"""
        currentRow = self.tableWidget.currentRow()
        # print 'Row %d' % currentRow
        if currentRow > -1:
            self.tableWidget.removeRow(currentRow)
            # print 'Before deletion'
            # print self.configuration['FiberSets'][currentRow]
            del self.configuration['FiberSets'][currentRow]
            # print 'After deletion'
            # print self.configuration['FiberSets'][currentRow]

    def generationChanged(self):
        """ Disables periodicity selection for potential generation """

        active = self.radioButton_PotGen.isChecked()

        if active:
            # print 'Disable periodicity'
            self.radioButton_fullPeriod.setChecked(active)
        # else:
        #     print 'Enable periodicity'

        self.radioButton_fullPeriod.setEnabled(not active)
        self.radioButton_vertPeriod.setEnabled(not active)
        self.radioButton_horizPeriod.setEnabled(not active)
        self.radioButton_noPeriod.setEnabled(not active)

    def compactionChanged(self):
        """ Enables radio buttons for compaction methods """

        active = self.checkBox_compaction.isChecked()
        # print 'Compaction is ' + str(active)

        # Group box
        self.groupBox_compaction.setEnabled(active)
        # Compaction - Point
        self.radioButton_compPoint.setEnabled(active)
        # Compaction - Direction
        self.radioButton_compDirection.setEnabled(active)
        # Compaction - Stirring
        self.radioButton_compStir.setEnabled(active)
        # Compaction - Box2D
        self.radioButton_compBox2D.setEnabled(active)

        if not active:
            self.algorithm['COMPACTION'] = None

        # Compaction - Point
        activePoint = self.radioButton_compPoint.isChecked()
        self.label_compPoint.setEnabled(activePoint)
        self.lineEdit_pointx.setEnabled(activePoint)
        self.lineEdit_pointy.setEnabled(activePoint)
        # if activePoint:
        #     # self.algorithm['COMPACTION'] = (RVE.POINT, (0.0, 0.0))
        #     print 'Point compaction'

        # Compaction - Direction
        activeDirection = self.radioButton_compDirection.isChecked()
        self.label_compDirection.setEnabled(activeDirection)
        self.lineEdit_vectorx.setEnabled(activeDirection)
        self.lineEdit_vectory.setEnabled(activeDirection)
        # if activeDirection:
        #     # self.algorithm['COMPACTION'] = (RVE.DIRECTION, (0.0, 0.0))
        #     print 'Direction compaction'

        # Compaction - Stir
        activeStir = self.radioButton_compStir.isChecked()
        self.label_compStir.setEnabled(activeStir)
        self.doubleSpinBox_compStir.setEnabled(activeStir)
        # if activeDirection:
        #     # self.algorithm['COMPACTION'] = (RVE.STIRRING, 100.0)
        #     print 'Stirring compaction'

        # # Compaction - Box2D
        activeBox2D = self.radioButton_compBox2D.isChecked()
        self.label_gravity.setEnabled(activeBox2D)
        self.lineEdit_gravx.setEnabled(activeBox2D)
        self.lineEdit_gravy.setEnabled(activeBox2D)
        self.checkBox_shake.setEnabled(activeBox2D)
        # if activeBos2D:
        #     print 'Physical compaction'

    def accept(self):  # Overriding method into this class
        """
        Exceptions are defined: NullSize, NoFiberSets, NullTolerance
        """
        # Size dimension cannot be equal to 0
        class NullSize(Exception): pass
        # At least one fiber set must be defined
        class NoFiberSets(Exception): pass
        # Tolerance cannot be equal to 0
        class NullTolerance(Exception): pass
        # Periodic generation (square or hexagonal) only support circular fibers
        class PeriodicGenerationError(Exception): pass
        # If compDirection is enabled, vector cannot be Null (0,0)
        class ZeroVector(Exception): pass
        # Potential generation of non-circular fibres
        # class PotentialTODO(Exception): pass

        # Update RVE size and tolerance
        self.sizeRve = (self.spinBox_SizeX.value(), self.spinBox_SizeY.value())
        self.configuration['Size'] = self.sizeRve
        self.configuration['Tolerance'] = self.spinBox_tolerance.value()

        # Generation algorithm
        if self.radioButton_randomGen.isChecked():
            self.algorithm['GENERATION'] = RVE.RANDOM
        elif self.radioButton_NNAgen.isChecked():
            self.algorithm['GENERATION'] = RVE.NNA
        elif self.radioButton_periodicHexagonalGen.isChecked():
            self.algorithm['GENERATION'] = '-'.join([RVE.PERIODIC, RVE.HEXAGONAL])
        elif self.radioButton_periodicSquareGen.isChecked():
            self.algorithm['GENERATION'] = '-'.join([RVE.PERIODIC, RVE.SQUARE])
        elif self.radioButton_PotGen.isChecked():
            self.algorithm['GENERATION'] = RVE.POTENTIAL
        elif self.radioButton_DynGen.isChecked():
            self.algorithm['GENERATION'] = RVE.DYNAMIC

        # Periodicity option
        if self.radioButton_fullPeriod.isChecked():
            self.algorithm['PERIODICITY'] = RVE.FULL
        elif self.radioButton_horizPeriod.isChecked():
            self.algorithm['PERIODICITY'] = RVE.HORIZ
        elif self.radioButton_vertPeriod.isChecked():
            self.algorithm['PERIODICITY'] = RVE.VERT
        elif self.radioButton_noPeriod.isChecked():
            self.algorithm['PERIODICITY'] = RVE.NONE

        # Compaction algorithm
        if self.checkBox_compaction.isChecked():
            if self.radioButton_compPoint.isChecked():
                coord_x = float(self.lineEdit_pointx.text())
                coord_y = float(self.lineEdit_pointy.text())
                self.algorithm['COMPACTION'] = (RVE.POINT, (coord_x, coord_y))
            elif self.radioButton_compDirection.isChecked():
                coord_x = float(self.lineEdit_vectorx.text())
                coord_y = float(self.lineEdit_vectory.text())
                self.algorithm['COMPACTION'] = (RVE.DIRECTION, (coord_x, coord_y))
            elif self.radioButton_compStir.isChecked():
                effect = float(self.doubleSpinBox_compStir.value())/100.0
                self.algorithm['COMPACTION'] = (RVE.STIRRING, effect)
            elif self.radioButton_compBox2D.isChecked():
                grav_x = float(self.lineEdit_gravx.text())
                grav_y = float(self.lineEdit_gravy.text())
                shake = self.checkBox_shake.isChecked()
                self.algorithm['COMPACTION'] = (RVE.BOX2D, (grav_x, grav_y), shake)
            else:
                self.algorithm['COMPACTION'] = None
        else:
            self.algorithm['COMPACTION'] = ()

        # Final check
        try:
            if (self.sizeRve[0] == 0) or (self.sizeRve[1] == 0):
                raise NullSize, 'Microstructure dimension cannot be equal to 0'
            elif len(self.configuration['FiberSets']) == 0:
                raise NoFiberSets, 'Number of fiber sets cannot be equal to 0'
            elif self.spinBox_tolerance.value() == 0:
                raise NullTolerance, 'Tolerance cannot be equal to 0'
            elif (RVE.PERIODIC in self.algorithm['GENERATION']) and \
                    len(self.configuration['FiberSets'])>1:# or (self.configuration['FiberSets'][0]['Geometry'].upper() != RVE.CIRCULAR)):
                    # (len(self.configuration['FiberSets'])>1 or (self.configuration['FiberSets'][0]['Geometry'].upper() != RVE.CIRCULAR)):
                raise PeriodicGenerationError, "Periodic generation only supports ONE fiber set" # which must be 'Circular'"
            elif (self.algorithm['COMPACTION']==(RVE.DIRECTION,(0,0))):
                raise ZeroVector, "Directional compaction algorithm requires not null vector"
            elif (self.algorithm['COMPACTION']) and (self.algorithm['COMPACTION'][0] == RVE.BOX2D) and not all([fset[RVE.GEOMETRY].upper() == RVE.CIRCULAR for fset in self.configuration['FiberSets']]):
                raise OnlyCircular, "Gravitational compaction is only supported for circular fibers"
            elif self.algorithm['COMPACTION'] == (RVE.BOX2D, (0, 0), False):
                raise NullCompaction, "No compaction will be performed"
            # elif self.algorithm['GENERATION'] == RVE.POTENTIAL and \
            #         not all([fset[RVE.GEOMETRY].upper() == RVE.CIRCULAR for fset in self.configuration['FiberSets']]):
            #     raise PotentialTODO, "Warning! Non-circular fibres generation is not stable, but it is allowed"
            else:
                QDialog.accept(self)  # accept method from QDialog!

        # except PotentialTODO, e:
        #     QMessageBox.warning(self, __appName__, str(e))
        #     QDialog.accept(self)

        except NullSize, e:
            QMessageBox.warning(self, __appName__, str(e))
            if self.spinBox_SizeX.value() == 0:
                self.spinBox_SizeX.selectAll()
                self.spinBox_SizeX.setFocus()
            else:
                self.spinBox_SizeY.selectAll()
                self.spinBox_SizeY.setFocus()
            return

        except NullTolerance, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.spinBox_tolerance.selectAll()
            self.spinBox_tolerance.setFocus()

        except NoFiberSets, e:
            QMessageBox.warning(self, __appName__, str(e))

        except PeriodicGenerationError, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.pushButton_remove.setFocus()

        except ZeroVector, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.lineEdit_vectorx.selectAll()
            self.lineEdit_vectorx.setFocus()

        except OnlyCircular, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.pushButton_remove.setFocus()

        except NullCompaction, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.radioButton_compBox2D.setFocus()


class AddFibers(NewMicrostructure):
    """ Add fibers to the current microstructure """

    def __init__(self, microstructure, parent=None):
        super(AddFibers, self).__init__(parent)
        self.setWindowTitle('Add Fibers Dialog')

        # Disable 'algorithm' tab
        # self.tab_algorithm.setEnabled(False)
        self.tabWidget.setCurrentIndex(0)
        # self.tabWidget.setCurrentWidget(self.tab_geometry)

        # Fix tolerance
        self.spinBox_tolerance.setEnabled(False)
        self.spinBox_tolerance.setProperty("value", microstructure.tolerance)

        # Fix RVE size
        self.spinBox_SizeX.setEnabled(False)
        self.spinBox_SizeY.setEnabled(False)
        self.spinBox_SizeX.setProperty("value", microstructure.rveSize[0])
        self.spinBox_SizeY.setProperty("value", microstructure.rveSize[1])

        # Disable regular arrangements (square and hexagonal)
        self.radioButton_periodicHexagonalGen.setEnabled(False)
        self.radioButton_periodicSquareGen.setEnabled(False)


class LobularAddParams(QDialog, addParamLobular.Ui_AddParamLobular):
    """ Dialog for additional geometrical parameters of lobular fibers """

    def __init__(self, parent=None):
        super(LobularAddParams, self).__init__(parent)
        self.setupUi(self)
        self.parameters = [self.spinBox_lobes.value()]
        # Connections
        self.spinBox_lobes.valueChanged.connect(self.lobes_number_changed)
        self.pushButton_ok.clicked.connect(self.accept)

    def lobes_number_changed(self):
        self.parameters = [self.spinBox_lobes.value()]


class EllipticalAddParams(QDialog, addParamElli.Ui_AddParamElli):
    """ Dialog for additional geometrical parameters of elliptical fibers """

    def __init__(self, parent=None):
        super(EllipticalAddParams, self).__init__(parent)
        self.setupUi(self)
        self.parameters = [self.spinBox_eccentricity.value()]
        # Connections
        self.spinBox_eccentricity.valueChanged.connect(self.eccentricity_changed)
        self.pushButton_ok.clicked.connect(self.accept)

    def eccentricity_changed(self):
        self.parameters = [self.spinBox_eccentricity.value()]


class PolygonAddParams(QDialog, addParamPoly.Ui_AddParamPolygonal):
    """ Dialog for additional geometrical parameters of polygonal fibers """
    def __init__(self, parent=None):
        super(PolygonAddParams, self).__init__(parent)
        self.setupUi(self)
        self.parameters = [self.spinBox_edges.value()]
        # Connections
        self.spinBox_edges.valueChanged.connect(self.edges_changed)
        self.pushButton_ok.clicked.connect(self.accept)

    def edges_changed(self):
        self.parameters[0] = self.spinBox_edges.value()


class SpolygonAddParams(QDialog, addParamSpoly.Ui_AddParamPoly):
    """ Dialog for additional geometrical parameters of smoothed polygonal fibers """

    def __init__(self, parent=None):
        super(SpolygonAddParams, self).__init__(parent)
        self.setupUi(self)
        self.parameters = [self.spinBox_edges.value(), self.spinBox_smoothRatio.value()]
        # Connections
        self.spinBox_edges.valueChanged.connect(self.edges_changed)
        self.spinBox_smoothRatio.valueChanged.connect(self.smoothRatio_changed)
        self.pushButton_ok.clicked.connect(self.accept)

    def edges_changed(self):
        self.parameters[0] = self.spinBox_edges.value()

    def smoothRatio_changed(self):
        self.parameters[1] = self.spinBox_smoothRatio.value()


class OvalAddParams(QDialog, addParamOval.Ui_AddParamOval):
    """ Dialog for additional geometrical parameters of smoothed polygonal fibers """

    def __init__(self, parent=None):
        super(OvalAddParams, self).__init__(parent)
        self.setupUi(self)
        self.parameters = [self.spinBox_slenderness.value(), self.spinBox_sharpness.value()]
        # Connections
        self.spinBox_slenderness.valueChanged.connect(self.slenderness_changed)
        self.spinBox_sharpness.valueChanged.connect(self.sharpness_changed)
        self.pushButton_ok.clicked.connect(self.accept)

    def slenderness_changed(self):
        self.parameters[0] = self.spinBox_slenderness.value()

    def sharpness_changed(self):
        self.parameters[1] = self.spinBox_sharpness.value()


class CshapeAddParams(QDialog, addParamCshape.Ui_AddParamCshape):
    """ Dialog for additional geometrical parameters of smoothed polygonal fibers """

    def __init__(self, parent=None):
        super(CshapeAddParams, self).__init__(parent)
        self.setupUi(self)
        self.parameters = [self.doubleSpinBox_chi.value(), self.dial.value()]
        # Connections
        self.pushButton_ok.clicked.connect(self.accept)

    def accept(self):  # Overriding method into this class
        """
        Exception defined: SelfIntersection, NullAngle
        """
        # Geometry self-intersects
        class SelfIntersection(Exception): pass
        # Null angle is not allowed
        class NullAngle(Exception): pass
        # Null hollowness is not allowed
        class NullHollowness(Exception): pass

        self.parameters[0] = chi = self.doubleSpinBox_chi.value()
        self.parameters[1] = theta = self.dial.value()

        try:
            if chi-1.0e-6 < 0:
                raise NullHollowness, 'Specify a valid hollowness'
            elif theta-1.0e-6 < 0:
                raise NullAngle, 'Specify a valid angle'
            elif (theta>180.) and ((np.sin(0.5*theta*np.pi/180) <= (1-chi)/(1+chi))):
                raise SelfIntersection, 'Current parameters define a self-intersecting geometry'
            else:
                QDialog.accept(self)  # accept method from QDialog!

        except NullHollowness, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.doubleSpinBox_chi.setFocus()
            self.doubleSpinBox_chi.selectAll()

        except NullAngle, e:
            QMessageBox.warning(self, __appName__, str(e))
            self.dial.setFocus()

        except SelfIntersection, e:
            QMessageBox.warning(self, __appName__, str(e))

###########################
## EXCEPTIONS
###########################

class ZeroVector(Exception): pass


class NullCompaction(Exception): pass

# Gravitational compaction is only supported for circular fibers
class OnlyCircular(Exception): pass


# Main execution of the Main Window
def main():
    QCoreApplication.setApplicationName(__appName__)
    QCoreApplication.setApplicationVersion(__appVersion__)
    QCoreApplication.setOrganizationName(__appName__)
    QCoreApplication.setOrganizationDomain(__appName__ + ".com")


    print "\n\n"
    print texting.superText(message='VIPER')
    # print " ####################################################"
    # print " ##                                                ##"
    # print " ##                    VIP3R                       ##"
    # print " ##                                                ##"
    # print " ####################################################\n\n"

    app = QApplication(sys.argv)
    form = Main()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()
