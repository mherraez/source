# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'addParamLobular.ui'
#
# Created: Fri Mar 20 11:09:21 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_AddParamLobular(object):
    def setupUi(self, AddParamLobular):
        AddParamLobular.setObjectName("AddParamLobular")
        AddParamLobular.resize(477, 718)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AddParamLobular.sizePolicy().hasHeightForWidth())
        AddParamLobular.setSizePolicy(sizePolicy)
        AddParamLobular.setMinimumSize(QtCore.QSize(477, 718))
        AddParamLobular.setMaximumSize(QtCore.QSize(477, 718))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/RVEs/lobular3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        AddParamLobular.setWindowIcon(icon)
        self.gridLayout = QtGui.QGridLayout(AddParamLobular)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtGui.QLabel(AddParamLobular)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.spinBox_lobes = QtGui.QSpinBox(AddParamLobular)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_lobes.sizePolicy().hasHeightForWidth())
        self.spinBox_lobes.setSizePolicy(sizePolicy)
        self.spinBox_lobes.setMinimum(2)
        self.spinBox_lobes.setMaximum(100)
        self.spinBox_lobes.setProperty("value", 3)
        self.spinBox_lobes.setObjectName("spinBox_lobes")
        self.horizontalLayout.addWidget(self.spinBox_lobes)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem1 = QtGui.QSpacerItem(20, 30, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtGui.QLabel(AddParamLobular)
        self.label_2.setMaximumSize(QtCore.QSize(455, 575))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/icons/RVEs/lobular3_plan.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem2 = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.pushButton_ok = QtGui.QPushButton(AddParamLobular)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.horizontalLayout_3.addWidget(self.pushButton_ok)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)

        self.retranslateUi(AddParamLobular)
        QtCore.QMetaObject.connectSlotsByName(AddParamLobular)

    def retranslateUi(self, AddParamLobular):
        AddParamLobular.setWindowTitle(QtGui.QApplication.translate("AddParamLobular", "Lobular fibers", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("AddParamLobular", "Number of lobes:", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_ok.setText(QtGui.QApplication.translate("AddParamLobular", "Ok", None, QtGui.QApplication.UnicodeUTF8))

import icons_rc
