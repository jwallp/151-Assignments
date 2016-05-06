import numpy as np


class Householder:
    """
    This class computes Householder transformations on an input matrix
    """

    def __init__(self, tableau):
        self.tableau = tableau

    def householder_reflection(self, sub_dim):
        """ Follows http://www.math.sjsu.edu/~foster/m143m/least_squares_via_Householder.pdf """

        if np.allclose(self.tableau, np.triu(self.tableau)):
            return self.tableau

        # Perform householder transformation on sub_m
        sub_m = self.tableau[sub_dim:, sub_dim:]

        z = self.tableau[sub_dim:,sub_dim]
        z0_sign = -1 * np.sign(z[0,0])
        z_norm = np.linalg.norm(z)

        e = [0]*len(z)
        e[0] = 1
        e = np.asmatrix(e).transpose()

        # v = sign(z0)||z||e - z and then v = v/||v||
        v = np.asmatrix(np.subtract(np.dot(z0_sign*z_norm, e), z))
        v = np.divide(v, np.linalg.norm(v))

        # new sub_m = sub_m - 2*v(vT_sub_m)
        vT_sub_m = np.dot(v.transpose(), sub_m)
        sub_m = np.subtract(sub_m, np.dot(2*v, vT_sub_m))

        # recombine sub-matrix into main matrix
        self.tableau[sub_dim:, sub_dim:] = sub_m

        return self.tableau

    def get_R(self):
        R = None
        for i in range(min(self.tableau.shape[0]-1, self.tableau.shape[1])):
            R = self.householder_reflection(i)
        return R

    def back_solve(self):
        m = self.tableau.shape[0]
        n = self.tableau.shape[1]
        x = np.empty(n-1)

        for i in range(n-2, -1, -1):
            xi = self.tableau[i, n-1]

            for j in range(i+1, n-1):
                xi = xi - self.tableau[i, j] * x[j]

            # this if-statement is needed for the abalone dataset because of
            # the binary proxy variables for sex can cause divide-by-zero errors
            if self.tableau[i, i] == 0.:
                # TODO: how to handle this case?
                x[i] = 0.
                # x[i] = xi
            else:
                x[i] = xi / self.tableau[i, i]

        return x

    def regression_prediction(self, dataset):
        #Get number of features of the dataset
        n = dataset.shape[1]

        w = self.back_solve()
        # multiply the input dataset (which is expected to contain in its
        # last column its actual classification) with the vector of weights
        return dataset[:, 0:n-1].dot(w)

    def get_coefficient(self):
        n = self.tableau.shape[1]
        return self.tableau[:, 0:n-1]

    def get_b(self):
        n = self.tableau.shape[1]
        return self.tableau[:, n-1:n]