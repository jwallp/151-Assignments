import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten

import random as rand
import SampleWithoutReplacement as swr
import KMeans
import logging

np.set_printoptions(threshold='nan')

RAND_NUM = 777

features = np.array([[1.9, 2.3],
                     [1.5, 2.5],
                     [0.8, 0.6],
                     [0.4, 1.8],
                     [0.1, 0.1],
                     [0.2, 1.8],
                     [2.0, 0.5],
                     [0.3, 1.5],
                     [1.0, 1.0]])
normalized = whiten(features)
numpy_result = kmeans(normalized, 2)

our_result = KMeans.k_means(normalized.tolist(), 2)
print numpy_result
print our_result
