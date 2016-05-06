from unittest import TestCase
import numpy as np
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
        self.assertTrue(np.allclose(result, [0.76, 0.6]), msg="FAIL: Backsolve T1")

        file_name = 'datasets/adjusted-abalone.csv'
        sampler = swr.SampleWithoutReplacement(file_name, .10)
        sampler.select()

        training_set = np.mat(sampler.get_training_set())
        trainer = hh.Householder(training_set)
        trainer.get_R() #QR decomposition

        our_backsolve = trainer.back_solve()
        numpy_backsolve = np.linalg.lstsq(trainer.get_coefficient(), trainer.get_b())[0]
        np.savetxt("testfiles/our_backsolve.csv", np.asarray(our_backsolve), delimiter=",")
        np.savetxt("testfiles/numpy_backsolve.csv", np.asarray(numpy_backsolve), delimiter=",")

        print our_backsolve
        print numpy_backsolve
        self.assertTrue(np.allclose(our_backsolve, numpy_backsolve),
                        msg="FAIL: Backsolve T2")


