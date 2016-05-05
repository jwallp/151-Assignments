import numpy as np


class Householder:
    """
    This class computes Householder transformations on an input matrix
    """

    def __init__(self, tableau):
        self.tableau = tableau

    def householder_reflection(self, sub_dim):
        if np.allclose(self.tableau, np.triu(self.tableau)):
            return self.tableau

        sub_m = self.tableau[sub_dim:, sub_dim:]
        z = self.tableau[sub_dim:,sub_dim]

        z0_sign = -1 * np.sign(z[0,0])
        z_norm2 = np.linalg.norm(z)

        e = [0]*len(z)
        e[0] = 1
        e = np.asmatrix(e).transpose()

        v = np.asmatrix(np.subtract(np.dot(z0_sign*z_norm2, e), z))
        #write-up uses subtract. Ours used add. Values now align with test.
        #v = np.asmatrix(np.add(np.dot(z0_sign*z_norm2, e), z))
        v = np.divide(v, np.linalg.norm(v))

        vt = v.getT()
        vtm = np.dot(vt, sub_m)
        two_v = np.dot(2, v)
        two_v_vtm = np.dot(two_v, vtm)
        sub_m = np.subtract(sub_m, two_v_vtm)
        self.tableau[sub_dim:, sub_dim:] = sub_m

        return self.tableau

    def get_R(self):
        r = None

        for i in range(min(self.tableau.shape[0]-1, self.tableau.shape[1])):
            r = self.householder_reflection(i)

        return r

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

    # TODO: how to do this entire method?
    def regression_prediction(self, dataset):
        # m = dataset.shape[0]
        n = dataset.shape[1]

        w = self.back_solve()
        # multiply the input dataset (which is expected to contain in its
        # last column its actual classification) with the vector of weights
        return dataset[:, 0:n-1].dot(w)


    def regression_predictionb(self, dataset):
        # m = dataset.shape[0]
        n = dataset.shape[1]

        w = self.back_solveb()
        # multiply the input dataset (which is expected to contain in its
        # last column its actual classification) with the vector of weights
        return dataset[:, 0:n - 1].dot(w)


    def back_solveb(self):
        m = self.tableau.shape[0]
        n = self.tableau.shape[1]
        x = np.empty(n - 1)

        for i in range(n - 2, -1, -1):
            xi = self.tableau[i, n - 1]

            for j in range(i + 1, n - 1):
                xi = xi - self.tableau[i, j] * x[j]

            # this if-statement is needed for the abalone dataset because of
            # the binary proxy variables for sex can cause divide-by-zero errors
            if self.tableau[i, i] == 0.:
                # TODO: how to handle this case?
                x[i] = 500.
                # x[i] = xi
            else:
                x[i] = xi / self.tableau[i, i]

        return x