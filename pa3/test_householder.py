from unittest import TestCase
import numpy as np
import Householder as hh


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

        expected = -1 * np.mat([[-1.6536, -1.1405, -1.2569],
                                [0, 0.9661, 0.6341], [0, 0, -0.8816], [0, 0, 0],
                                [0, 0, 0]])
        our_result = self.householder.get_R()
        print expected
        print our_result

        print np.isclose(expected, our_result)
        self.assertTrue(np.allclose(our_result, expected),
                        msg="FAIL: Householder's implementation")

        # print self.householder.get_R()
        # print self.householder.back_solve()

    def testBackSolve(self):
        print "Testing back solving..."

        r = np.mat([[5., 2., 5.], [0., 5., 3.], [0., 0., -4.]])
        h2 = hh.Householder(r)

        print h2.back_solve()

