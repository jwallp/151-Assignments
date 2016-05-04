import numpy as np



def HouseHolders(m, sub_dim):
    if np.allclose(m, np.triu(m)):
        return m

    sub_m = m[sub_dim:, sub_dim:]
    print(sub_m)
    z = m[sub_dim:,sub_dim]
    #print(z)

    z0_sign = -1 * np.sign(z[0,0])
    z_norm2 = np.linalg.norm(z)
    #print(z_norm2)

    e = [0]*len(z)
    e[0] = 1
    e = np.asmatrix(e).transpose()
    #print(e)

    v = np.asmatrix(np.add(np.dot(z0_sign*z_norm2, e), z))
    v = np.divide(v, np.linalg.norm(v))
    #print (v)

    vt = v.getT()
    vtm = np.dot(vt, sub_m)
    two_v = np.dot(2, v)
    two_v_vtm = np.dot(two_v, vtm)
    sub_m = np.subtract(sub_m, two_v_vtm)
    print(sub_m)
    m[sub_dim:, sub_dim:] = sub_m
    print(m)
    return m


#m = np.matrix([[1, -1.0, 1.0], [1, -0.5, 0.25], [1, 0.0, 0.0], [1, 0.5, 0.25], [1, 1.0, 1.0]])
m = np.matrix([[0.8147, 0.0975, 0.1576], [0.9058, 0.2785, 0.9706], [0.1270, 0.5469, 0.9572], [0.9134, 0.9575, 0.4854], [0.6324, 0.9649, 0.8003]])
for i in range(m.shape[0]):
    print(i, " = K ..................")
    m = HouseHolders(m, i)
