#!/usr/bin/env python

import os
import pickle
from tqdm import tqdm
from PIL import Image
from ocrd_typegroups_classifier.typegroups_classifier import TypegroupsClassifier

tgc = TypegroupsClassifier.load('ocrd_typegroups_classifier/models/classifier.tgc')

with open('descr.dat', 'wb') as f:
    for fname in tqdm(os.listdir('subset')):
        fname = os.path.join('subset', fname)
        im    = Image.open(fname).convert('RGB')
        features, classification = tgc.describe(im, batch_size=48, stride=112)
        pickle.dump((fname, features, classification), f)
