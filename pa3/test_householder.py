from unittest import TestCase
import numpy as np
import random as rand
np.set_printoptions(threshold='nan')
import csv

import Householder as hh
import SampleWithoutReplacement as swr


class TestHouseholder(TestCase):
    def setUp(self):
        # TODO: try using different example
        self.A = np.mat([[0.8147, 0.0975, 0.1576], [0.9058, 0.2785, 0.9706],
                         [0.1270, 0.5469, 0.9572], [0.9134, 0.9575, 0.4854],
                         [0.6324, 0.9649, 0.8003]])
        self.householder = hh.Householder(self.A)


    # TODO: add assertions for test
    def testHH(self):
        print "Testing Householder reflections..."


        print "Test 1 -------------------------------------------------------"
        expected = np.mat([[-1.65365, -1.14046, -1.25697],
                                [0, 0.96609, 0.634107], [0, 0, -0.88155], [0, 0, 0],
                                [0, 0, 0]])
        our_result = self.householder.get_R()
        print expected
        print our_result
        self.assertTrue(np.allclose(our_result, expected),
                        msg="FAIL: Householder's implementation T1")


        print "Test 2 -------------------------------------------------------"
        householder_test = hh.Householder(np.mat([
            [1.0,-1.0,-1.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 1.0],
            [2.0,-2.0, 1.0],
            [3.0, 2.0, 1.0]
        ]))
        expected = np.mat([
            [-4.36, -1.15, -2.06],
            [    0, -3.56, -1.58],
            [    0,     0,   2.5],
            [    0,     0,     0],
            [    0,     0,     0]
        ])
        our_result = householder_test.get_R()
        print expected
        print our_result

        print "Test 3 -------------------------------------------------------"
        householder_test = hh.Householder(np.mat([
            [3.0, -2.0, 3.0],
            [0.0, 3.0, 5.0],
            [4.0, 4.0, 4.0],
        ]))
        expected = np.mat([
            [-5, -2, -5],
            [ 0, -5, -3],
            [ 0,  0, -4],
        ])
        our_result = householder_test.get_R()
        print expected
        print our_result

    def testBackSolve(self):
        print "Testing back solving..."

        r = np.mat([[5., 2., 5.], [0., 5., 3.], [0., 0., -4.]])
        h2 = hh.Householder(r)

        result = h2.back_solve()
        numpy_backsolve = np.linalg.lstsq(h2.get_coefficient(), h2.get_b())[0]

        self.assertTrue(np.allclose(np.asmatrix(result).transpose(), numpy_backsolve), msg="FAIL: Backsolve T1")

    def testAbaloneRMSE(self):
        print "Testing abalone data set RMSE..."

        #sample data set without sex columns
        rand.seed(777)
        sampler = swr.SampleWithoutReplacement('datasets/adjusted-abalone.csv', .10)
        sampler.select()

        test_set1 = np.mat(sampler.get_test_set())
        trainer = hh.Householder(np.mat(sampler.get_training_set()))
        trainer.get_R() #QR decomposition
        our_backsolve_without_sex = trainer.back_solve()

        #sample data set with sex columns
        rand.seed(777)
        sampler2 = swr.SampleWithoutReplacement('datasets/adjusted-abalone2.csv', .10)
        sampler2.select()

        test_set2 = np.mat(sampler.get_test_set())
        trainer2 = hh.Householder(np.mat(sampler2.get_training_set()))
        trainer2.get_R() #QR decomposition
        numpy_backsolve_with_sex = np.linalg.lstsq(trainer2.get_coefficient(), trainer2.get_b())[0]

        #BACKSOLVE obviously produces different results between the two.
        np.savetxt("testfiles/our_backsolve.csv", np.asarray(our_backsolve_without_sex), delimiter=",")
        np.savetxt("testfiles/numpy_backsolve.csv", np.asarray(numpy_backsolve_with_sex), delimiter=",")

        print"RMSE---------------------------------------------"
        #RMSE of the dataset without sex columns
        predictions = trainer.regression_prediction(test_set1)
        actual = test_set1[:, -1].T
        difference = predictions - actual

        rmse = np.sqrt(difference.dot(difference.T)[0, 0] / test_set1.shape[0])
        print "\t->RMSE ours without sex columns=%s" % rmse

        #RMSE of the dataset with sex columns
        predictions2 = trainer.regression_prediction2(test_set2).T
        actual = test_set2[:, -1].T
        difference = predictions2 - actual

        rmse2 = np.sqrt(difference.dot(difference.T)[0, 0] / test_set2.shape[0])
        print "\t->RMSE numpy with sex columns=%s" % rmse2

        self.assertTrue(np.allclose(rmse, rmse2), msg="FAIL: RMSE Abalone test")
        