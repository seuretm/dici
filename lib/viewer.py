from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtWidgets import QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
import sys
import random
import numpy
from PIL import Image
from PIL.ImageQt import ImageQt
from sklearn import mixture
from sklearn import cluster

# Note for code readers: the more you read, the more obvious it will be
# that this was written by somebody 

class ProjView(QMainWindow):
    def __init__(self, raw_data, proj_data, display_info, DataDisplayClass):
        super().__init__()
        self.proj_data = proj_data
        self.raw_data = raw_data
        self.display_info = display_info
        self.initUI(DataDisplayClass)
        self.plot.plot(proj_data)
        
    def initUI(self, DataDisplayClass):
        self.setWindowTitle('Prosy - projection to sample viewer')
        wid = QWidget(self)
        self.setCentralWidget(wid)
 
        hb = QHBoxLayout()
        self.plot = PlotCanvas(self, width=5, height=4)
        hb.addWidget(self.plot)
        
        self.data_display = DataDisplayClass(self)
        hb.addWidget(self.data_display)
        
        wid.setLayout(hb)
        
        menu = self.menuBar()
        gmmMenu = menu.addMenu('Fit projection')
        
        btn = QAction('K-Means', self)
        btn.triggered.connect(self.compute_kmeans)
        gmmMenu.addAction(btn)
        
        btn = QAction('Gaussian Mixture', self)
        btn.triggered.connect(self.compute_gmm)
        gmmMenu.addAction(btn)
        
        btn = QAction('Bayesian Gaussian Mixture', self)
        btn.triggered.connect(self.compute_bgmm)
        gmmMenu.addAction(btn)
        
        btn = QAction('Dirichlet Process Gaussian Mixture Model', self)
        btn.triggered.connect(self.compute_dpgmm)
        gmmMenu.addAction(btn)
        
        gmmMenu = menu.addMenu('Fit raw data')
        
        btn = QAction('K-Means', self)
        btn.triggered.connect(self.compute_kmeans_raw)
        gmmMenu.addAction(btn)
        
        btn = QAction('Gaussian Mixture', self)
        btn.triggered.connect(self.compute_gmm_raw)
        gmmMenu.addAction(btn)
        
        btn = QAction('Bayesian Gaussian Mixture', self)
        btn.triggered.connect(self.compute_bgmm_raw)
        gmmMenu.addAction(btn)
        
        btn = QAction('Dirichlet Process Gaussian Mixture Model', self)
        btn.triggered.connect(self.compute_dpgmm_raw)
        gmmMenu.addAction(btn)
        
        
        self.showMaximized()

    def askNbClusters(self):
        i, okPressed = QInputDialog.getInt(self, "Enter an integer value","Number of clusters:", 3, 2, 100, 1)
        if okPressed:
            return i
        return -1
        
    def compute_kmeans(self):
        c = self.askNbClusters()
        gmm = cluster.KMeans(n_clusters=c).fit(self.proj_data)
        pre = gmm.predict(self.proj_data)
        self.plot.plot(self.proj_data, pre)
    
    def compute_gmm(self):
        c = self.askNbClusters()
        gmm = mixture.GaussianMixture(n_components=c, covariance_type='full').fit(self.proj_data)
        pre = gmm.predict(self.proj_data)
        self.plot.plot(self.proj_data, pre)
    
    def compute_bgmm(self):
        c = self.askNbClusters()
        gmm = mixture.BayesianGaussianMixture(n_components=c, covariance_type='full').fit(self.proj_data)
        pre = gmm.predict(self.proj_data)
        self.plot.plot(self.proj_data, pre)
    
    def compute_dpgmm(self):
        gmm = mixture.BayesianGaussianMixture(n_components=100, covariance_type='full').fit(self.proj_data)
        pre = gmm.predict(self.proj_data)
        self.plot.plot(self.proj_data, pre)
    
    def compute_kmeans_raw(self):
        c = self.askNbClusters()
        gmm = cluster.KMeans(n_clusters=c).fit(self.raw_data)
        pre = gmm.predict(self.raw_data)
        self.plot.plot(self.proj_data, pre)
    
    def compute_gmm_raw(self):
        c = self.askNbClusters()
        gmm = mixture.GaussianMixture(n_components=c, covariance_type='full').fit(self.raw_data)
        pre = gmm.predict(self.raw_data)
        self.plot.plot(self.proj_data, pre)
    
    def compute_bgmm_raw(self):
        c = self.askNbClusters()
        gmm = mixture.BayesianGaussianMixture(n_components=c, covariance_type='full').fit(self.raw_data)
        pre = gmm.predict(self.raw_data)
        self.plot.plot(self.proj_data, pre)
    
    def compute_dpgmm_raw(self):
        gmm = mixture.BayesianGaussianMixture(n_components=100, covariance_type='full').fit(self.raw_data)
        pre = gmm.predict(self.raw_data)
        self.plot.plot(self.proj_data, pre)

class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.parent = parent
        self.closest = None
        self.highlighted = None
        
        self.data = None
        self.labels = None
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.fig.canvas.mpl_connect(s='button_press_event', func=self.onclick)
    
    def onclick(self, event):
        # event.dblclick, event.button, event.x, event.y, event.xdata, event.ydata
        if self.data is None:
            return
        
        if not self.highlighted is None:
            self.highlighted.remove()
        
        pt = [event.xdata, event.ydata]
        self.closest = numpy.argmin(numpy.sum((self.data - pt)**2, 1))
        self.parent.data_display.display(self.parent.display_info[self.closest])
        self.highlighted = self.axes.scatter(self.data[self.closest,0], self.data[self.closest,1], color='r', s=12)
        self.draw()
        
        
    def plot(self, data, labels=None):
        self.data = data
        self.labels = labels
        if self.labels is None:
            self.axes.set_title('Projection view')
            self.axes.scatter(data[:,0], data[:,1], color='b', s=4)
        else:
            self.axes.cla()
            self.axes.set_title('Projection and clustering view')
            for lbl in set(self.labels):
                self.axes.scatter(data[self.labels==lbl,0], data[self.labels==lbl,1], s=4)
            
        
        self.draw()

class DataDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def display(self, data):
        raise Exception('Not implemented')

class TextDisplay(DataDisplay):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.le = QLineEdit()
        hb = QHBoxLayout()
        hb.addWidget(self.le)
        self.setLayout(hb)

    def display(self, data):
        self.le.setText(data)

class ImageDisplay(DataDisplay):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.filepath = QLineEdit()
        hb = QVBoxLayout()
        hb.addWidget(self.filepath)
        hb.addWidget(self.scrollArea)
        self.setLayout(hb)

    def display(self, data):
        self.filepath.setText(data)
        pm = QPixmap(data)
        self.imageLabel.setPixmap(pm)
        self.imageLabel.resize(pm.width(), pm.height())



















