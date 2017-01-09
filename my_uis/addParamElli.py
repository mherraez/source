# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'addParamElli.ui'
#
# Created: Fri Mar 20 11:09:30 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_AddParamElli(object):
    def setupUi(self, AddParamElli):
        AddParamElli.setObjectName("AddParamElli")
        AddParamElli.resize(426, 708)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AddParamElli.sizePolicy().hasHeightForWidth())
        AddParamElli.setSizePolicy(sizePolicy)
        AddParamElli.setMaximumSize(QtCore.QSize(426, 708))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/RVEs/ellipsis.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        AddParamElli.setWindowIcon(icon)
        self.verticalLayout_2 = QtGui.QVBoxLayout(AddParamElli)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtGui.QLabel(AddParamElli)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem = QtGui.QSpacerItem(35, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.spinBox_eccentricity = QtGui.QDoubleSpinBox(AddParamElli)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_eccentricity.sizePolicy().hasHeightForWidth())
        self.spinBox_eccentricity.setSizePolicy(sizePolicy)
        self.spinBox_eccentricity.setMinimumSize(QtCore.QSize(20, 0))
        self.spinBox_eccentricity.setMaximum(1.0)
        self.spinBox_eccentricity.setSingleStep(0.1)
        self.spinBox_eccentricity.setProperty("value", 0.75)
        self.spinBox_eccentricity.setObjectName("spinBox_eccentricity")
        self.horizontalLayout_2.addWidget(self.spinBox_eccentricity)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        spacerItem1 = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.label_2 = QtGui.QLabel(AddParamElli)
        self.label_2.setMinimumSize(QtCore.QSize(408, 580))
        self.label_2.setMaximumSize(QtCore.QSize(408, 580))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/icons/RVEs/ellipse_plan.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        spacerItem2 = QtGui.QSpacerItem(20, 19, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.verticalLayout_2.addItem(spacerItem2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.pushButton_ok = QtGui.QPushButton(AddParamElli)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.horizontalLayout.addWidget(self.pushButton_ok)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(AddParamElli)
        QtCore.QMetaObject.connectSlotsByName(AddParamElli)

    def retranslateUi(self, AddParamElli):
        AddParamElli.setWindowTitle(QtGui.QApplication.translate("AddParamElli", "Elliptical fibers", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("AddParamElli", "Eccentricity:", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_ok.setText(QtGui.QApplication.translate("AddParamElli", "Ok", None, QtGui.QApplication.UnicodeUTF8))

import icons_rc
