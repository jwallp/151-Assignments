import random as rand


def read_user_input(threshold, datalist, counts, N):
    Nr = N
    Nn = round(threshold * N)

    if counts is None:
        counts = [0] * N

    return threshold_test(datalist, counts, N, Nn, Nr)


def threshold_test(datalist, counts, N, Nn, Nr):
    while Nn > 0:
        draw = rand.random()
        if draw < Nn / Nr:
            counts[N - Nr] += 1
            Nn -= 1
        Nr -= 1
    return counts
