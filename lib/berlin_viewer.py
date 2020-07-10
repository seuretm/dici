#!/usr/bin/env python

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPalette
from PyQt5 import QtCore
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QLayoutItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QPushButton
import sys
import random
import numpy
from PIL import Image
from PIL.ImageQt import ImageQt
import urllib
from viewer import DataDisplay
import cv2
from PyQt5.QtWidgets import QAction
import numpy
import pickle
from PIL import Image
import sys
from ocrd_typegroups_classifier.typegroups_classifier import TypegroupsClassifier

class BerlinImageDisplay(DataDisplay):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.groundTruth = QLabel('  ')
        self.filepath = QLineEdit()
        hb = QVBoxLayout()
        hb.addWidget(self.filepath)
        hb.addWidget(self.groundTruth)
        self.fitCB = QCheckBox("Fit to screen",self)
        self.fitCB.stateChanged.connect(self.redraw)
        self.classifyBtn = QPushButton('Run classifier')
        self.classifyBtn.clicked.connect(self.classify)
        hb.addWidget(self.classifyBtn)
        hb.addWidget(self.fitCB)
        hb.addWidget(self.scrollArea)
        self.layoutItem = hb.itemAt(4)
        self.setLayout(hb)
        self.currentData = None
        self.label = {}
        for x in open('multiclass.csv', 'rt'):
            spl = x.strip().split(',')
            self.label[spl[0]] = spl[1:]
        self.tgc = None
    
    def redraw(self):
        if self.currentData is None:
            return
        self.display(self.currentData)
    
    def classify(self):
        if self.currentData is None:
            return
        self.classifyBtn.setEnabled(False)
        print('Loading the image')
        im = Image.open(self.currentData)
        print('Loading the classifier')
        if self.tgc is None:
            self.tgc = TypegroupsClassifier.load('ocrd_typegroups_classifier/models/classifier.tgc')
        print('Starting the classification')
        res = self.tgc.classify(im, stride=129, batch_size=12, score_as_key=True)
        dsp = 'Classification:'
        for s in sorted(res,reverse=True):
            if s>0:
                dsp = '%s\n%15s: %.2f' % (dsp, res[s], s)
        QMessageBox.information(None, 'Results', dsp, QMessageBox.Ok)
        self.classifyBtn.setEnabled(True)
        

    def display(self, data):
        self.currentData = data
        self.filepath.setText(data)
        img = cv2.imread(data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qimg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        vert_ratio = pm.height() / float(self.layoutItem.geometry().height())
        hori_ratio = pm.width() / float(self.layoutItem.geometry().width())
        if self.fitCB.checkState()== QtCore.Qt.Checked:
            ratio = max(vert_ratio, hori_ratio)
        else:
            ratio = 1
        pm = pm.scaled(int(pm.width()/ratio), int(pm.height()/ratio))
        self.imageLabel.setPixmap(pm)
        self.imageLabel.resize(pm.width(), pm.height())
        
        gts = self.label[data][0]
        for x in self.label[data][1:]:
            gts = '%s, %s' % (gts, x)
        self.groundTruth.setText(gts)
    
    def custom_menu(self):
        return None
        btn = QAction('Map data', self)
        btn.triggered.connect(self.map_data)
        return btn
    
    def map_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select data file", "","Dat Files (*.dat);;All Files (*)", options=options)
        if not fileName:
            return
        bf = open(fileName, 'rb')
        labels   = list()
        filepath = list()
        scores   = list()
        features = list()
        while True:
            try:
                labels.append(pickle.load(bf))
                filepath.append(pickle.load(bf))
                scores.append(pickle.load(bf))
                features.append(pickle.load(bf))
            except:
                break
        bf.close()
        print('data loaded')
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
