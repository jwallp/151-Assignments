import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import random as rand
import SampleWithoutReplacement as swr
import KMeans
from unittest import TestCase
np.set_printoptions(threshold='nan')


class TestKMeans(TestCase):
    def test1(self):
        print "TEST 1:----------------------------------------------------------------"
        features = np.array([[1.9, 2.3],
                          [1.5, 2.5],
                          [0.8, 0.6],
                          [0.4, 1.8],
                          [0.1, 0.1],
                          [0.2, 1.8],
                          [2.0, 0.5],
                          [0.3, 1.5],
                          [1.0, 1.0]])
        whitened = whiten(features)
        book = np.array((whitened[0], whitened[2]))
        numpy_result = kmeans(whitened, book)[0]
        print numpy_result
        print ""

        features2 = np.array([[1.9, 2.3,0],
                             [1.5, 2.5,0],
                             [0.8, 0.6,0],
                             [0.4, 1.8,0],
                             [0.1, 0.1,0],
                             [0.2, 1.8,0],
                             [2.0, 0.5,0],
                             [0.3, 1.5,0],
                             [1.0, 1.0,0]])
        whitened2 = whiten(features2)
        book2 = [whitened[0], whitened[2]]
        our_result = np.array(KMeans.k_means2(whitened2.tolist(), 2, book2).centroids)[:, :-1]
        print our_result

        #self.assertTrue(np.allclose(numpy_result, our_result),
        #                msg="FAIL: KMeans do not match for hardcoded array K=2.")

    def test2(self):
        print "TEST 2:----------------------------------------------------------------"
        rand.seed(777)
        sampler = swr.SampleWithoutReplacement('datasets/adjusted-abalone.csv', .10)
        sampler.z_scale()
        training_set = sampler.get_training_set()
        test_set = sampler.get_test_set()

        indices_selected = list()
        centroids = [None]*4
        for i in range(4):
            while True:
                index_selected = np.random.randint(0, len(training_set))
                if index_selected not in indices_selected:
                    centroids[i] = training_set[index_selected]
                    indices_selected.append(index_selected)
                    break

        numpy_result=kmeans(np.array(training_set)[:, :-1], np.array(centroids)[:, :-1])[0]
        our_result=np.array(KMeans.k_means2(training_set, 4, centroids).centroids)[:, :-1]
        print numpy_result
        print ""
        print our_result
        #self.assertTrue(np.allclose(numpy_result, our_result),
        #                msg="FAIL: KMeans do not match for adjusted-abalone K=4.")




