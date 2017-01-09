# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plotSettings.ui'
#
# Created: Fri Mar 20 00:59:00 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_plotSettings(object):
    def setupUi(self, plotSettings):
        plotSettings.setObjectName("plotSettings")
        plotSettings.resize(198, 166)
        self.gridLayout = QtGui.QGridLayout(plotSettings)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.checkBox_colour = QtGui.QCheckBox(plotSettings)
        self.checkBox_colour.setEnabled(True)
        font = QtGui.QFont()
        font.setWeight(50)
        font.setBold(False)
        self.checkBox_colour.setFont(font)
        self.checkBox_colour.setAutoFillBackground(False)
        self.checkBox_colour.setChecked(True)
        self.checkBox_colour.setTristate(False)
        self.checkBox_colour.setObjectName("checkBox_colour")
        self.verticalLayout.addWidget(self.checkBox_colour)
        self.checkBox_numbered = QtGui.QCheckBox(plotSettings)
        self.checkBox_numbered.setObjectName("checkBox_numbered")
        self.verticalLayout.addWidget(self.checkBox_numbered)
        self.checkBox_ticks = QtGui.QCheckBox(plotSettings)
        self.checkBox_ticks.setChecked(True)
        self.checkBox_ticks.setObjectName("checkBox_ticks")
        self.verticalLayout.addWidget(self.checkBox_ticks)
        self.checkBox_COM = QtGui.QCheckBox(plotSettings)
        self.checkBox_COM.setObjectName("checkBox_COM")
        self.verticalLayout.addWidget(self.checkBox_COM)
        self.checkBox_MOI = QtGui.QCheckBox(plotSettings)
        self.checkBox_MOI.setObjectName("checkBox_MOI")
        self.verticalLayout.addWidget(self.checkBox_MOI)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        spacerItem = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        self.verticalLayout_2.addItem(spacerItem)
        self.buttonBox = QtGui.QDialogButtonBox(plotSettings)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_2.addWidget(self.buttonBox)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        self.retranslateUi(plotSettings)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), plotSettings.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), plotSettings.reject)
        QtCore.QMetaObject.connectSlotsByName(plotSettings)

    def retranslateUi(self, plotSettings):
        plotSettings.setWindowTitle(QtGui.QApplication.translate("plotSettings", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_colour.setText(QtGui.QApplication.translate("plotSettings", "Colour", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_numbered.setText(QtGui.QApplication.translate("plotSettings", "Numbered fibres", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_ticks.setText(QtGui.QApplication.translate("plotSettings", "Show ticks and title", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_COM.setText(QtGui.QApplication.translate("plotSettings", "Center of Mass", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_MOI.setText(QtGui.QApplication.translate("plotSettings", "Moments of Inertia", None, QtGui.QApplication.UnicodeUTF8))

