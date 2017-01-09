# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'localVolumeFraction.ui'
#
# Created: Mon May 30 17:16:33 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_localVf(object):
    def setupUi(self, localVf):
        localVf.setObjectName("localVf")
        localVf.resize(477, 546)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(localVf.sizePolicy().hasHeightForWidth())
        localVf.setSizePolicy(sizePolicy)
        localVf.setMinimumSize(QtCore.QSize(477, 500))
        localVf.setMaximumSize(QtCore.QSize(477, 800))
        self.gridLayout = QtGui.QGridLayout(localVf)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.pushButton_ok = QtGui.QPushButton(localVf)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.horizontalLayout_3.addWidget(self.pushButton_ok)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtGui.QLabel(localVf)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem1 = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.spinBox_spotSize = QtGui.QDoubleSpinBox(localVf)
        self.spinBox_spotSize.setMaximum(99999.99)
        self.spinBox_spotSize.setProperty("value", 20.0)
        self.spinBox_spotSize.setObjectName("spinBox_spotSize")
        self.horizontalLayout.addWidget(self.spinBox_spotSize)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_2 = QtGui.QLabel(localVf)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_5.addWidget(self.label_2)
        spacerItem2 = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.spinBox_spotSpacing = QtGui.QDoubleSpinBox(localVf)
        self.spinBox_spotSpacing.setMaximum(99999.99)
        self.spinBox_spotSpacing.setProperty("value", 1.0)
        self.spinBox_spotSpacing.setObjectName("spinBox_spotSpacing")
        self.horizontalLayout_5.addWidget(self.spinBox_spotSpacing)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        spacerItem3 = QtGui.QSpacerItem(20, 30, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.verticalLayout.addItem(spacerItem3)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.figure_localVf = QtGui.QLabel(localVf)
        self.figure_localVf.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.figure_localVf.sizePolicy().hasHeightForWidth())
        self.figure_localVf.setSizePolicy(sizePolicy)
        self.figure_localVf.setMaximumSize(QtCore.QSize(1200, 1200))
        self.figure_localVf.setFrameShape(QtGui.QFrame.NoFrame)
        self.figure_localVf.setText("")
        self.figure_localVf.setPixmap(QtGui.QPixmap(":/icons/RVEs/localVf.png"))
        self.figure_localVf.setAlignment(QtCore.Qt.AlignCenter)
        self.figure_localVf.setObjectName("figure_localVf")
        self.horizontalLayout_2.addWidget(self.figure_localVf)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(localVf)
        QtCore.QMetaObject.connectSlotsByName(localVf)

    def retranslateUi(self, localVf):
        localVf.setWindowTitle(QtGui.QApplication.translate("localVf", "Local fibre volume fraction", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_ok.setText(QtGui.QApplication.translate("localVf", "Ok", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("localVf", "Spot size (S_R)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("localVf", "Spot spacing (S_s)", None, QtGui.QApplication.UnicodeUTF8))

import icons_rc
