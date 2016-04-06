import random as rand
import csv


def read_user_input(threshold, counts):
    with open('abalone.csv') as csvfile:
        reader = csv.reader(csvfile)

        datalist = list(reader)
        N = len(datalist)
        Nr = N
        Nn = round(threshold * N)

        if counts is None:
            counts = [0] * N

        return threshold_test(counts, N, Nn, Nr)


def threshold_test(counts, N, Nn, Nr):
    while Nn > 0:
        draw = rand.random()
        if draw < Nn/Nr:
            counts[N-Nr] += 1
            Nn -= 1
        Nr -= 1

    return counts

        #return counts

"""
        for row in reader:
            draw = rng.random()
            if draw < Nn/Nr:
                counts[]
                """
