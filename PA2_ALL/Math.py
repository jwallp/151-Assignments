import math
from collections import Counter
import sys


def euclidean_distance(observ1, observ2):
    result = 0.0
    for i in range(len(observ1) - 1):
        result += pow((observ1[i] - observ2[i]), 2)
    return math.sqrt(result)


def mode( data):
    counter = Counter(data)
    max_count = max(counter.values())
    mode = [k for k, v in counter.items() if v == max_count]
    return mode[0]


def max_of_lists(list_of_sets):
    max_val = 0
    for subset in list_of_sets:
        for row in subset:
            if row[len(subset[0]) - 1] > max_val:
                max_val = row[len(subset[0]) - 1]
    return int(max_val)


def min_of_lists(list_of_sets):
    min_val = sys.maxsize
    for subset in list_of_sets:
        for row in subset:
            if row[len(subset[0]) - 1] < min_val:
                min_val = row[len(subset[0]) - 1]
    return int(min_val)