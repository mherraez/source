# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compactSettings.ui'
#
# Created: Thu May 07 10:22:57 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(547, 184)
        self.verticalLayout = QtGui.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtGui.QGroupBox(Dialog)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtGui.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_pointy = QtGui.QLineEdit(self.groupBox)
        self.lineEdit_pointy.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lineEdit_pointy.setObjectName("lineEdit_pointy")
        self.gridLayout.addWidget(self.lineEdit_pointy, 0, 4, 1, 1)
        self.radioButton_compDirection = QtGui.QRadioButton(self.groupBox)
        self.radioButton_compDirection.setEnabled(True)
        self.radioButton_compDirection.setObjectName("radioButton_compDirection")
        self.gridLayout.addWidget(self.radioButton_compDirection, 1, 0, 1, 1)
        spacerItem = QtGui.QSpacerItem(10, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
        self.doubleSpinBox_compStir = QtGui.QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_compStir.setEnabled(False)
        self.doubleSpinBox_compStir.setMaximumSize(QtCore.QSize(60, 16777215))
        self.doubleSpinBox_compStir.setDecimals(1)
        self.doubleSpinBox_compStir.setProperty("value", 100.0)
        self.doubleSpinBox_compStir.setObjectName("doubleSpinBox_compStir")
        self.gridLayout.addWidget(self.doubleSpinBox_compStir, 2, 3, 1, 1)
        self.lineEdit_vectory = QtGui.QLineEdit(self.groupBox)
        self.lineEdit_vectory.setEnabled(False)
        self.lineEdit_vectory.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lineEdit_vectory.setObjectName("lineEdit_vectory")
        self.gridLayout.addWidget(self.lineEdit_vectory, 1, 4, 1, 1)
        self.radioButton_compStir = QtGui.QRadioButton(self.groupBox)
        self.radioButton_compStir.setEnabled(True)
        self.radioButton_compStir.setObjectName("radioButton_compStir")
        self.gridLayout.addWidget(self.radioButton_compStir, 2, 0, 1, 1)
        self.label_compStir = QtGui.QLabel(self.groupBox)
        self.label_compStir.setEnabled(False)
        self.label_compStir.setObjectName("label_compStir")
        self.gridLayout.addWidget(self.label_compStir, 2, 2, 1, 1)
        self.radioButton_compPoint = QtGui.QRadioButton(self.groupBox)
        self.radioButton_compPoint.setChecked(True)
        self.radioButton_compPoint.setObjectName("radioButton_compPoint")
        self.gridLayout.addWidget(self.radioButton_compPoint, 0, 0, 1, 1)
        self.lineEdit_vectorx = QtGui.QLineEdit(self.groupBox)
        self.lineEdit_vectorx.setEnabled(False)
        self.lineEdit_vectorx.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lineEdit_vectorx.setObjectName("lineEdit_vectorx")
        self.gridLayout.addWidget(self.lineEdit_vectorx, 1, 3, 1, 1)
        self.lineEdit_pointx = QtGui.QLineEdit(self.groupBox)
        self.lineEdit_pointx.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lineEdit_pointx.setObjectName("lineEdit_pointx")
        self.gridLayout.addWidget(self.lineEdit_pointx, 0, 3, 1, 1)
        self.label_compPoint = QtGui.QLabel(self.groupBox)
        self.label_compPoint.setObjectName("label_compPoint")
        self.gridLayout.addWidget(self.label_compPoint, 0, 2, 1, 1)
        self.label_compDirection = QtGui.QLabel(self.groupBox)
        self.label_compDirection.setEnabled(False)
        self.label_compDirection.setObjectName("label_compDirection")
        self.gridLayout.addWidget(self.label_compDirection, 1, 2, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(60, 20, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 5, 1, 1)
        self.radioButton_compBox2D = QtGui.QRadioButton(self.groupBox)
        self.radioButton_compBox2D.setObjectName("radioButton_compBox2D")
        self.gridLayout.addWidget(self.radioButton_compBox2D, 3, 0, 1, 1)
        self.label_gravity = QtGui.QLabel(self.groupBox)
        self.label_gravity.setEnabled(False)
        self.label_gravity.setObjectName("label_gravity")
        self.gridLayout.addWidget(self.label_gravity, 3, 2, 1, 1)
        self.lineEdit_gravx = QtGui.QLineEdit(self.groupBox)
        self.lineEdit_gravx.setEnabled(False)
        self.lineEdit_gravx.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lineEdit_gravx.setObjectName("lineEdit_gravx")
        self.gridLayout.addWidget(self.lineEdit_gravx, 3, 3, 1, 1)
        self.lineEdit_gravy = QtGui.QLineEdit(self.groupBox)
        self.lineEdit_gravy.setEnabled(False)
        self.lineEdit_gravy.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lineEdit_gravy.setObjectName("lineEdit_gravy")
        self.gridLayout.addWidget(self.lineEdit_gravy, 3, 4, 1, 1)
        self.checkBox_shake = QtGui.QCheckBox(self.groupBox)
        self.checkBox_shake.setEnabled(False)
        self.checkBox_shake.setObjectName("checkBox_shake")
        self.gridLayout.addWidget(self.checkBox_shake, 3, 5, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Dialog", "Compaction technique", None, QtGui.QApplication.UnicodeUTF8))
        self.lineEdit_pointy.setText(QtGui.QApplication.translate("Dialog", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_compDirection.setText(QtGui.QApplication.translate("Dialog", "Compact in a direction", None, QtGui.QApplication.UnicodeUTF8))
        self.lineEdit_vectory.setText(QtGui.QApplication.translate("Dialog", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_compStir.setText(QtGui.QApplication.translate("Dialog", "Stirring algorithm", None, QtGui.QApplication.UnicodeUTF8))
        self.label_compStir.setText(QtGui.QApplication.translate("Dialog", "Effectiveness (%):", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_compPoint.setText(QtGui.QApplication.translate("Dialog", "Compact towards a point", None, QtGui.QApplication.UnicodeUTF8))
        self.lineEdit_vectorx.setText(QtGui.QApplication.translate("Dialog", "1.0", None, QtGui.QApplication.UnicodeUTF8))
        self.lineEdit_pointx.setText(QtGui.QApplication.translate("Dialog", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.label_compPoint.setText(QtGui.QApplication.translate("Dialog", "Point coordinates:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_compDirection.setText(QtGui.QApplication.translate("Dialog", "Vector components:", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_compBox2D.setText(QtGui.QApplication.translate("Dialog", "Physical compaction", None, QtGui.QApplication.UnicodeUTF8))
        self.label_gravity.setText(QtGui.QApplication.translate("Dialog", "Gravity:", None, QtGui.QApplication.UnicodeUTF8))
        self.lineEdit_gravx.setText(QtGui.QApplication.translate("Dialog", "0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.lineEdit_gravy.setText(QtGui.QApplication.translate("Dialog", "-10.0", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox_shake.setText(QtGui.QApplication.translate("Dialog", "Shake", None, QtGui.QApplication.UnicodeUTF8))

