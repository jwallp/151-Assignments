import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten

import random as rand
import SampleWithoutReplacement as swr
import KMeans
import logging

np.set_printoptions(threshold='nan')

rand.seed(777)
sampler = swr.SampleWithoutReplacement('datasets/adjusted-abalone.csv', .10)
arr = np.array(sampler.get_training_set())
print np.linalg.lstsq(arr[:, :-1], arr[:, -1])[0]

rand.seed(777)
sampler2 = swr.SampleWithoutReplacement('datasets/adjusted-abalone.csv', .10)
sampler2.z_scale()
arr2 = np.array(sampler2.get_training_set())
print np.linalg.lstsq(arr2[:, :-1], arr2[:, -1])[0]
