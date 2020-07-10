from PyQt5.QtWidgets import QApplication
from viewer import ProjView
import numpy
import pickle

from viewer import ImageDisplay
from viewer import TextDisplay
from berlin_viewer import BerlinImageDisplay


def load_data(filename):
    bf = open(filename, 'rb')
    labels   = list()
    filepath = list()
    scores   = list()
    while True:
        try:
            labels.append(pickle.load(bf))
            filepath.append(pickle.load(bf))
            scores.append(pickle.load(bf))
        except:
            break
    bf.close()
    scores = numpy.array(scores)
    return labels, filepath, scores

_, _, raw = load_data('classification.dat')
labels, filepath, proj = load_data('tsne-p-120.dat')
# It's mandatory for the projection to be 2D
proj = proj[:, 0:2]


app = QApplication([])
pv = ProjView(raw, proj, filepath, BerlinImageDisplay)
app.exec()

