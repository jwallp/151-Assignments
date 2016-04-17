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


def max_of_lists(list1, list2):
    max_val = 0
    for row in list1:
        if row[len(list1[0]) - 1] > max_val:
            max_val = row[len(list1[0]) - 1]
    for row in list2:
        if row[len(list2[0]) - 1] > max_val:
            max_val = row[len(list2[0]) - 1]
    return int(max_val)


def min_of_lists(list1, list2):
    min_val = sys.maxsize
    for row in list1:
        if row[len(list1[0]) - 1] < min_val:
            min_val = row[len(list1[0]) - 1]
    for row in list2:
        if row[len(list2[0]) - 1] < min_val:
            min_val = row[len(list2[0]) - 1]
    return int(min_val)