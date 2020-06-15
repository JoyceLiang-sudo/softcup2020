# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


import sys

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtWidgets import QWidget, QApplication


class Ui_Form(QWidget):
    def __init__(self):
        super(Ui_Form, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.close_flag = True

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1139, 833)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QtCore.QSize(8, 8))
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.show_video = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(4)
        sizePolicy.setVerticalStretch(8)
        sizePolicy.setHeightForWidth(self.show_video.sizePolicy().hasHeightForWidth())
        self.show_video.setSizePolicy(sizePolicy)
        self.show_video.setMinimumSize(QtCore.QSize(10, 10))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.show_video.setFont(font)
        self.show_video.setFrameShape(QtWidgets.QFrame.Box)
        self.show_video.setObjectName("show_video")
        self.gridLayout.addWidget(self.show_video, 2, 1, 4, 1)
        self.label = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.show_message = QtWidgets.QTextBrowser(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_message.sizePolicy().hasHeightForWidth())
        self.show_message.setSizePolicy(sizePolicy)
        self.show_message.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        self.show_message.setObjectName("show_message")
        self.gridLayout.addWidget(self.show_message, 2, 0, 2, 1)
        self.show_message2 = QtWidgets.QTextBrowser(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_message2.sizePolicy().hasHeightForWidth())
        self.show_message2.setSizePolicy(sizePolicy)
        self.show_message2.setObjectName("show_message2")
        self.gridLayout.addWidget(self.show_message2, 4, 0, 2, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.show_video.setText(_translate("Form", "展示检测结果"))
        self.label.setText(_translate("Form", "\t\t\t\t交通场景智能应用\t\t\t\t"))

    def closeEvent(self, event):
        self.close_flag = False
