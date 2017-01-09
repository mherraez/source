# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'addParamSpoly.ui'
#
# Created: Thu Oct 02 01:48:04 2014
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_AddParamPoly(object):
    def setupUi(self, AddParamPoly):
        AddParamPoly.setObjectName("AddParamPoly")
        AddParamPoly.resize(194, 123)
        self.gridLayout = QtGui.QGridLayout(AddParamPoly)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtGui.QLabel(AddParamPoly)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_2.addLayout(self.horizontalLayout_3)
        self.spinBox_smoothRatio = QtGui.QDoubleSpinBox(AddParamPoly)
        self.spinBox_smoothRatio.setMaximum(1.0)
        self.spinBox_smoothRatio.setSingleStep(0.1)
        self.spinBox_smoothRatio.setProperty("value", 0.5)
        self.spinBox_smoothRatio.setObjectName("spinBox_smoothRatio")
        self.horizontalLayout_2.addWidget(self.spinBox_smoothRatio)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtGui.QLabel(AddParamPoly)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.spinBox_edges = QtGui.QSpinBox(AddParamPoly)
        self.spinBox_edges.setMinimum(3)
        self.spinBox_edges.setMaximum(100)
        self.spinBox_edges.setProperty("value", 3)
        self.spinBox_edges.setObjectName("spinBox_edges")
        self.horizontalLayout.addWidget(self.spinBox_edges)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.pushButton_ok = QtGui.QPushButton(AddParamPoly)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.horizontalLayout_4.addWidget(self.pushButton_ok)
        self.gridLayout.addLayout(self.horizontalLayout_4, 3, 0, 1, 1)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 2, 0, 1, 1)

        self.retranslateUi(AddParamPoly)
        QtCore.QMetaObject.connectSlotsByName(AddParamPoly)

    def retranslateUi(self, AddParamPoly):
        AddParamPoly.setWindowTitle(QtGui.QApplication.translate("AddParamPoly", "Spolygon fibers", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("AddParamPoly", "Smoothing ratio:", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("AddParamPoly", "Number of edges:", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_ok.setText(QtGui.QApplication.translate("AddParamPoly", "Ok", None, QtGui.QApplication.UnicodeUTF8))

