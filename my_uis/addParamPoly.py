# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'addParamPoly.ui'
#
# Created: Tue Oct 07 21:07:12 2014
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_AddParamPolygonal(object):
    def setupUi(self, AddParamPolygonal):
        AddParamPolygonal.setObjectName("AddParamPolygonal")
        AddParamPolygonal.resize(200, 100)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AddParamPolygonal.sizePolicy().hasHeightForWidth())
        AddParamPolygonal.setSizePolicy(sizePolicy)
        self.gridLayout = QtGui.QGridLayout(AddParamPolygonal)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtGui.QLabel(AddParamPolygonal)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.spinBox_edges = QtGui.QSpinBox(AddParamPolygonal)
        self.spinBox_edges.setMinimum(3)
        self.spinBox_edges.setProperty("value", 4)
        self.spinBox_edges.setObjectName("spinBox_edges")
        self.horizontalLayout_2.addWidget(self.spinBox_edges)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 1, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.pushButton_ok = QtGui.QPushButton(AddParamPolygonal)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.horizontalLayout.addWidget(self.pushButton_ok)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 1)

        self.retranslateUi(AddParamPolygonal)
        QtCore.QMetaObject.connectSlotsByName(AddParamPolygonal)

    def retranslateUi(self, AddParamPolygonal):
        AddParamPolygonal.setWindowTitle(QtGui.QApplication.translate("AddParamPolygonal", "Polygonal fibers", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("AddParamPolygonal", "Number of edges:", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_ok.setText(QtGui.QApplication.translate("AddParamPolygonal", "Ok", None, QtGui.QApplication.UnicodeUTF8))

