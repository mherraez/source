# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'addParamOval.ui'
#
# Created: Thu Oct 02 01:47:51 2014
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_AddParamOval(object):
    def setupUi(self, AddParamOval):
        AddParamOval.setObjectName("AddParamOval")
        AddParamOval.resize(196, 116)
        self.gridLayout = QtGui.QGridLayout(AddParamOval)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtGui.QLabel(AddParamOval)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.spinBox_slenderness = QtGui.QDoubleSpinBox(AddParamOval)
        self.spinBox_slenderness.setMinimum(1.0)
        self.spinBox_slenderness.setMaximum(100.0)
        self.spinBox_slenderness.setObjectName("spinBox_slenderness")
        self.horizontalLayout_2.addWidget(self.spinBox_slenderness)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtGui.QLabel(AddParamOval)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.spinBox_sharpness = QtGui.QDoubleSpinBox(AddParamOval)
        self.spinBox_sharpness.setMinimum(0.0)
        self.spinBox_sharpness.setMaximum(1.0)
        self.spinBox_sharpness.setSingleStep(0.1)
        self.spinBox_sharpness.setProperty("value", 0.5)
        self.spinBox_sharpness.setObjectName("spinBox_sharpness")
        self.horizontalLayout_3.addWidget(self.spinBox_sharpness)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        spacerItem2 = QtGui.QSpacerItem(20, 8, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 2, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.pushButton_ok = QtGui.QPushButton(AddParamOval)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.horizontalLayout.addWidget(self.pushButton_ok)
        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 1)

        self.retranslateUi(AddParamOval)
        QtCore.QMetaObject.connectSlotsByName(AddParamOval)

    def retranslateUi(self, AddParamOval):
        AddParamOval.setWindowTitle(QtGui.QApplication.translate("AddParamOval", "Oval fibers", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("AddParamOval", "Slenderness:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("AddParamOval", "Sharpness:", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_ok.setText(QtGui.QApplication.translate("AddParamOval", "Ok", None, QtGui.QApplication.UnicodeUTF8))

