import numpy as np


class Householder:
    """
    This class computes Householder transformations on an input matrix
    """

    def __init__(self, tableau):
        self.tableau = tableau
        self.Aprime = tableau
        # self.m = self.tableau.shape[0]
        # self.n = self.tableau.shape[1]

    def ek(self, k, n):
        e = np.zeros(n)
        e[k] = 0
        return e

    def getQk(self, k):
        """ k is in the range [0,n-1] """
        # x = self.Aprime[:, k]
        x = self.Aprime[:, 0]
        xnorm = np.sqrt(x.dot(x))
        # e = self.ek(k, x.shape[0])
        e = self.ek(0, x.shape[0])
        u = x - (xnorm * e)
        v = u / (np.sqrt(u.dot(u)))

        m = self.Aprime.shape[0]
        Qprime = np.identity(m) - 2 * np.outer(v, v)
        # Update Aprime to be the new submatrix produced by eliminating top row
        # and top column of result of this iteration of Householder's
        self.Aprime = Qprime.dot(self.Aprime)[range(1, m)][:, range(1, m)]

        """ special case if on the first iteration """
        if k == 0:
            return Qprime

            # I = np.mat(np.identity(k-1))
            # z = np.mat(np.zeros((k-1, k-1)))
            # Qk = np.bmat([[I, z], [z, np.mat(Qprime)]])
            #
            # return Qk
