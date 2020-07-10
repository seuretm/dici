#!/usr/bin/env python

import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

filename = []
features = []
classifi = []
with open('descr.dat', 'rb') as f:
    while True:
        try:
            fn, ft, cl = pickle.load(f)
            filename.append(fn)
            features.append(ft)
            classifi.append(cl)
        except:
            break


print('Here', len(filename))
proj = TSNE(n_components=2, perplexity=60, n_iter=5000).fit_transform(features)


with open('proj.dat', 'wb') as f:
    for i in range(len(proj)):
        pickle.dump((filename[i], features[i], proj[i], classifi[i]), f)

