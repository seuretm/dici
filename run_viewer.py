#!/usr/bin/env python

from PyQt5.QtWidgets import QApplication
from viewer import ProjView
import numpy
import pickle

from viewer import ImageDisplay
from lib.berlin_viewer import BerlinImageDisplay
from viewer import TextDisplay


#pickle.dump((filename[i], proj[i], classifi[i]), f)

fname = []
raw   = []
proj  = []
out   = []
with open('proj.dat', 'rb') as f_in:
    while True:
        try:
            f, r, p, c = pickle.load(f_in)
            fname.append(f)
            raw.append(r)
            proj.append(p)
            out.append(c)
        except:
            break
proj = numpy.array(proj)
print(len(proj))

app = QApplication([])
pv = ProjView(raw, proj, fname, BerlinImageDisplay)
app.exec()



quit()

def load_data(filename):
    bf = open(filename, 'rb')
    labels   = pickle.load(bf)
    filepath = pickle.load(bf)
    proj     = pickle.load(bf)
    raw      = pickle.load(bf)
    bf.close()
    proj = numpy.array(proj)
    return labels, filepath, proj, raw

labels, filepath, proj, raw = load_data('tsne-p-120.dat')
# It's mandatory for the projection to be 2D
proj = proj[:, 0:2]


app = QApplication([])
pv = ProjView(raw, proj, filepath, BerlinImageDisplay)
app.exec()

