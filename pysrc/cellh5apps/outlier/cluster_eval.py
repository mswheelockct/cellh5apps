import numpy
import sklearn
import sklearn.cluster
import sklearn.mixture

import cPickle
import pylab

data = cPickle.load(open('cluster_training_data.pkl','r'))
