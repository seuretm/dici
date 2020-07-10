#!/usr/bin/env python

from PIL import Image
import sys
from ocrd_typegroups_classifier.typegroups_classifier import TypegroupsClassifier

if len(sys.argv)!=2:
    print('Syntax:\n  python run_classification.py input_image');
    quit()

print('Loading the image')
im = Image.open(sys.argv[1])

print('Loading the classifier')
tgc = TypegroupsClassifier.load('ocrd_typegroups_classifier/models/classifier.tgc')

print('Starting the classification')
res = tgc.classify(im, stride=129, batch_size=12, score_as_key=True)

for s in sorted(res,reverse=True):
    print('%15s: %.2f' % (res[s], s))
