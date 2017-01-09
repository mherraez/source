# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'addParamCshape.ui'
#
# Created: Wed Mar 11 17:08:13 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_AddParamCshape(object):
    def setupUi(self, AddParamCshape):
        AddParamCshape.setObjectName("AddParamCshape")
        AddParamCshape.resize(189, 160)
        self.gridLayout = QtGui.QGridLayout(AddParamCshape)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 2, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtGui.QLabel(AddParamCshape)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.doubleSpinBox_chi = QtGui.QDoubleSpinBox(AddParamCshape)
        self.doubleSpinBox_chi.setMaximum(1.0)
        self.doubleSpinBox_chi.setSingleStep(0.05)
        self.doubleSpinBox_chi.setProperty("value", 0.5)
        self.doubleSpinBox_chi.setObjectName("doubleSpinBox_chi")
        self.horizontalLayout.addWidget(self.doubleSpinBox_chi)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.pushButton_ok = QtGui.QPushButton(AddParamCshape)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.horizontalLayout_3.addWidget(self.pushButton_ok)
        self.gridLayout.addLayout(self.horizontalLayout_3, 4, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.dial = QtGui.QDial(AddParamCshape)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dial.sizePolicy().hasHeightForWidth())
        self.dial.setSizePolicy(sizePolicy)
        self.dial.setMaximum(360)
        self.dial.setProperty("value", 180)
        self.dial.setObjectName("dial")
        self.horizontalLayout_2.addWidget(self.dial)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.label_2 = QtGui.QLabel(AddParamCshape)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.label_arc = QtGui.QLabel(AddParamCshape)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_arc.sizePolicy().hasHeightForWidth())
        self.label_arc.setSizePolicy(sizePolicy)
        self.label_arc.setObjectName("label_arc")
        self.horizontalLayout_2.addWidget(self.label_arc)
        spacerItem4 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)

        self.retranslateUi(AddParamCshape)
        QtCore.QObject.connect(self.dial, QtCore.SIGNAL("valueChanged(int)"), self.label_arc.setNum)
        QtCore.QMetaObject.connectSlotsByName(AddParamCshape)

    def retranslateUi(self, AddParamCshape):
        AddParamCshape.setWindowTitle(QtGui.QApplication.translate("AddParamCshape", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("AddParamCshape", "Hollowness:", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_ok.setText(QtGui.QApplication.translate("AddParamCshape", "Ok", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("AddParamCshape", "Arc (º):", None, QtGui.QApplication.UnicodeUTF8))
        self.label_arc.setText(QtGui.QApplication.translate("AddParamCshape", "180", None, QtGui.QApplication.UnicodeUTF8))

