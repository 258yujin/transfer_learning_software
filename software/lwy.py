# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lwyui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets, QtCore, QtGui, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class Stream(QObject):
    """Redirects console output to text widget."""
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))
        QApplication.processEvents()

    def flush(self):
        pass
class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.mode = ""
        self.datafile = ""
        self.modelname = ""
        self.models = 'U-net'
        self.augment = False
        self.optimizers = 'Adam'
        self.method = 'Random Initialization'
        sys.stdout = Stream(newText=self.onUpdateText)
    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.process.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(973, 718)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_6.setContentsMargins(0, -1, -1, -1)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        # spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        # self.verticalLayout.addItem(spacerItem)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setStyleSheet("font: 75 15pt \"微软雅黑\";\n"
                                   "alternate-background-color: rgb(255, 255, 255);\n"
                                   "color: rgb(85, 170, 255);\n"
                                   "")
        self.label_8.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft)
        self.label_8.setObjectName("label_8")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem1)
        self.verticalLayout.addWidget(self.label_8, 0, QtCore.Qt.AlignTop)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setLabelAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.formLayout_2.setFormAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.formLayout_2.setContentsMargins(-1, 5, -1, -1)
        self.formLayout_2.setObjectName("formLayout_2")
        self.formLayout_2.setSpacing(18)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                        "alternate-background-color: rgb(255, 255, 255);\n"
                                        "")
        self.pushButton_3.setObjectName("pushButton_3")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                        "alternate-background-color: rgb(255, 255, 255);\n"
                                        "")
        self.pushButton_4.setObjectName("pushButton_4")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.pushButton_4)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                 "alternate-background-color: rgb(255, 255, 255);\n"
                                 "")
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                   "alternate-background-color: rgb(255, 255, 255);\n"
                                   "")
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.verticalLayout.addLayout(self.formLayout_2)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setStyleSheet("font: 75 15pt \"微软雅黑\";\n"
                                   "alternate-background-color: rgb(255, 255, 255);\n"
                                   "color: rgb(85, 170, 255);\n"
                                   "")
        self.label_9.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft)
        self.label_9.setObjectName("label_9")
        # self.formLayout_2 = QtWidgets.QFormLayout()
        # self.formLayout_2.setLabelAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.formLayout_2.setFormAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        # self.formLayout_2.setContentsMargins(-1, 5, -1, -1)
        # self.formLayout_2.setObjectName("formLayout_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                   "alternate-background-color: rgb(255, 255, 255);\n"
                                   "")
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_3.sizePolicy().hasHeightForWidth())
        self.comboBox_3.setSizePolicy(sizePolicy)
        self.comboBox_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.comboBox_3.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                        "alternate-background-color: rgb(255, 255, 255);\n"
                                        "")
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_3)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                   "alternate-background-color: rgb(255, 255, 255);\n"
                                   "")
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName("lineEdit")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                        "alternate-background-color: rgb(255, 255, 255);\n"
                                        "")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.comboBox)

        # self.comboBox_4 = QtWidgets.QComboBox(self.centralwidget)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.comboBox_4.sizePolicy().hasHeightForWidth())
        # self.comboBox_4.setSizePolicy(sizePolicy)
        # self.comboBox_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        # self.comboBox_4.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
        #                                 "alternate-background-color: rgb(255, 255, 255);\n"
        #                                 "")
        # self.comboBox_4.setObjectName("comboBox_4")
        # self.comboBox_4.addItem("")
        # self.comboBox_4.addItem("")
        # self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.comboBox_4)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                   "alternate-background-color: rgb(255, 255, 255);\n"
                                   "")
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                   "alternate-background-color: rgb(255, 255, 255);\n"
                                   "")
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
        self.checkBox.setSizePolicy(sizePolicy)
        self.checkBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.checkBox.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                    "alternate-background-color: rgb(255, 255, 255);\n"
                                    "")
        self.checkBox.setObjectName("checkBox")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.checkBox)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                   "alternate-background-color: rgb(255, 255, 255);\n"
                                   "")
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.verticalLayout.addLayout(self.formLayout_2)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setContentsMargins(0, -1, -1, 4)
        self.verticalLayout_2.setSpacing(7)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
                                        "alternate-background-color: rgb(255, 255, 255);\n"
                                        "")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.genMastClicked)
        # self.formLayout_2.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.pushButton_2)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.verticalLayout.addWidget(self.pushButton_2, 0, QtCore.Qt.AlignTop)
        self.img_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_label.sizePolicy().hasHeightForWidth())
        self.img_label.setSizePolicy(sizePolicy)
        self.img_label.setMinimumSize(QtCore.QSize(200, 0))
        self.img_label.setMaximumSize(QtCore.QSize(150, 150))
        self.img_label.setBaseSize(QtCore.QSize(0, 0))
        self.img_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.img_label.setText("")
        self.img_label.setPixmap(QtGui.QPixmap("pixmap.png"))
        self.img_label.setScaledContents(True)
        self.img_label.setAlignment(QtCore.Qt.AlignTop)
        self.img_label.setObjectName("img_label")
        self.verticalLayout_2.addWidget(self.img_label, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 2)
        self.verticalLayout.setStretch(3, 1)
        self.verticalLayout.setStretch(4, 3)
        self.verticalLayout.setStretch(5, 1)
        self.horizontalLayout_6.addLayout(self.verticalLayout)
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        # self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        # self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        # self.label_12 = QtWidgets.QLabel(self.centralwidget)
        # self.label_12.setStyleSheet("font: 75 10pt \"微软雅黑\";\n"
        #                            "alternate-background-color: rgb(255, 255, 255);\n"
        #                            "")
        # self.label_12.setObjectName("label_12")
        # self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_12)
        # self.pushButton_Run = QtWidgets.QPushButton(self.centralwidget)
        # self.pushButton_Run.setStyleSheet("font: 75 10pt \"微软雅黑\";")
        # self.pushButton_Run.setObjectName("pushButton_Run")
        # self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.pushButton_Run)
        # self.horizontalLayout_7.addWidget(self.pushButton_Run)
        # spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        # self.horizontalLayout_7.addItem(spacerItem2)
        # self.horizontalLayout_7.addWidget(self.pushButton_2)
        # spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        # self.horizontalLayout_7.addItem(spacerItem3)
        # self.verticalLayout_14.addLayout(self.horizontalLayout_7)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.graph1 = QtWidgets.QLabel(self.frame)
        self.graph1.setObjectName("label_10")
        self.gridLayout_3.addWidget(self.graph1, 0, 0, 1, 1)
        self.graph2 = QtWidgets.QLabel(self.frame)
        self.graph2.setObjectName("label_11")
        self.gridLayout_3.addWidget(self.graph2, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 1, 1, 1)
        self.verticalLayout_14.addLayout(self.gridLayout)
        self.process = QtWidgets.QTextEdit(self.centralwidget)
        self.process.setObjectName("textEdit")
        self.verticalLayout_14.addWidget(self.process)
        self.verticalLayout_14.setStretch(0, 1)
        self.verticalLayout_14.setStretch(1, 6)
        self.verticalLayout_14.setStretch(2, 3)
        self.horizontalLayout_6.addLayout(self.verticalLayout_14)
        self.horizontalLayout_6.setStretch(1, 9)
        self.gridLayout_4.addLayout(self.horizontalLayout_6, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionRun = QtWidgets.QAction(MainWindow)
        self.actionRun.setObjectName("actionRun")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_8.setText(_translate("MainWindow", "Experimental Setting"))
        self.pushButton_3.setText(_translate("MainWindow", "select"))
        self.pushButton_4.setText(_translate("MainWindow", "select"))
        self.label.setText(_translate("MainWindow", "Source domain"))
        self.label_2.setText(_translate("MainWindow", "Target domain"))
        # self.label_9.setText(_translate("MainWindow", "Experimental Setting"))
        self.label_3.setText(_translate("MainWindow", "model"))
        self.label_4.setText(_translate("MainWindow", "optimizer"))
        self.label_5.setText(_translate("MainWindow", "learning rate"))
        self.label_7.setText(_translate("MainWindow", "augmentation"))
        # self.label_12.setText(_translate("MainWindow", "Training method"))
        self.label_6.setText(_translate("MainWindow", "epoch"))
        # self.pushButton_Run.setText(_translate("MainWindow", "Random Initialization"))
        # self.pushButton_Run.clicked.connect(self.genMastClicked)
        self.pushButton_2.setText(_translate("MainWindow", "Run"))
        self.graph1.setText(_translate("MainWindow", "graph1"))
        self.graph2.setText(_translate("MainWindow", "graph2"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionRun.setText(_translate("MainWindow", "Run"))
        # self.comboBox_4.setItemText(0, _translate("MainWindow", "Random Initialization"))
        # self.comboBox_4.setItemText(1, _translate("MainWindow", "Fine-tune"))
        # self.comboBox_4.activated[str].connect(self.methodselect)
        self.comboBox_3.setItemText(0, _translate("MainWindow", "U-net"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "U-net++"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "U-net+++"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "deeplab v3"))
        self.comboBox_3.activated[str].connect(self.modelselect)
        self.comboBox.setItemText(0, _translate("MainWindow", "Adam"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Adamax"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Adagrad"))
        self.comboBox.setItemText(3, _translate("MainWindow", "RMSprop"))
        self.comboBox.setItemText(4, _translate("MainWindow", "SGD"))
        self.comboBox.activated[str].connect(self.optimizerselect)
        self.checkBox.stateChanged.connect(self.augmentation)

        self.pushButton_4.clicked.connect(self.msg)
        self.pushButton_3.clicked.connect(self.msg1)

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()

        pix = QPixmap("Figure_123.png")
        self.graph1.setPixmap(pix.scaled(640, 480, QtCore.Qt.KeepAspectRatio))
        self.graph1.setAlignment(Qt.AlignCenter)
        self.graph1.setScaledContents(True) # 设置图片自适应窗口大小

        pix = QPixmap("Figure_1234.png")
        self.graph2.setPixmap(pix.scaled(640, 480, QtCore.Qt.KeepAspectRatio))
        self.graph2.setAlignment(Qt.AlignCenter)
        self.graph2.setScaledContents(True) # 设置图片自适应窗口大小
    def msg(self, Filepath):
        d = QtWidgets.QFileDialog.getExistingDirectory(None, '选取文件夹', 'C:/')
        self.datafile = d
    def msg1(self, Filepath):
        d1 = QtWidgets.QFileDialog.getOpenFileName(self,  "选取文件", "C:/")
        self.modelname = d1
    def modelselect(self, text):
        self.models = text
    def optimizerselect(self, text):
        self.optimizers = text
    def methodselect(self, text):
        self.method = text
    def augmentation(self):
        if self.checkBox.isChecked():
            self.augment = True
        self.augment = False
    def genMastClicked(self):
        # """Runs the main function."""
        # print('Running...')
        # self.printhello()
        loop = QEventLoop()
        QTimer.singleShot(2000, loop.quit)
        # loop.exec_()
        # print('Done.')
        from train_predict2 import NestNet
        img_rows = 512
        img_cols = 512
        color_type = 2
        num_class = 2
        deep_supervision = False
        datafile = self.datafile
        if self.modelname == '':
            modelname = self.modelname
        else:
            modelname = self.modelname[0]
        epoch = self.lineEdit.text()
        lr = self.lineEdit_2.text()
        print('learning rate:', lr)
        print('epoch:', epoch)
        models = self.models
        augment = self.augment
        optimizers = self.optimizers
        method = self.method
        self.unet = NestNet(img_rows, img_cols, color_type, num_class, deep_supervision, method, datafile, modelname,
                            augment, models, epoch, lr, optimizers)
        self.unet.start()
        QApplication.processEvents()
        while self.unet.isRunning():
            pix = QPixmap("Figure_123.png")
            self.graph1.setPixmap(pix.scaled(640, 480, QtCore.Qt.KeepAspectRatio))
            self.graph1.setAlignment(Qt.AlignCenter)
            self.graph1.setScaledContents(True)  # 设置图片自适应窗口大小

            pix = QPixmap("Figure_1234.png")
            self.graph2.setPixmap(pix.scaled(640, 480, QtCore.Qt.KeepAspectRatio))
            self.graph2.setAlignment(Qt.AlignCenter)
            self.graph2.setScaledContents(True)  # 设置图片自适应窗口大小
            QApplication.processEvents()

    def fcku(self,fckimage):
        # hbox = QHBoxLayout(self)
        #print(fckimage.size())
        pil_image = self.m_resize(self.width(), self.height(), fckimage)
        # fckimage=cv2.cvtColor(fckimage,cv2.COLOR_RGB2BGR)
        #fckimage = QImage(fckimage.width, fckimage.height, QImage.Format_RGB888)
        # print(fckimage.width)

        pixmap = QPixmap.fromImage(pil_image)
        # print(pixmap.height())
        # pixmap = self.m_resize(self.width(), self.height(), pixmap)
        self.lbl.resize(pil_image.width(),pil_image.height())
        self.lbl.setPixmap(pixmap)
        #print(pixmap.size())
        # hbox.addWidget(lbl)
        # self.setLayout(hbox)

    def m_resize(self,w_box, h_box, pil_image):  # 参数是：要适应的窗口宽、高、Image.open后的图片

        w, h = pil_image.width(), pil_image.height() # 获取图像的原始大小

        f1 = 1.0*w_box/w
        f2 = 1.0 * h_box / h

        factor = min([f1, f2])

        width = int(w * factor)

        height = int(h * factor)
        #return pil_image.resize(width, height)
        return pil_image.scaled(width, height)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())