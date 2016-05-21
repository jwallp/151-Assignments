import collections
import numpy as np
import math
import sys


def k_means(training_set, k):
    clusters = [[] for i in range(k)]       # cluster[i] refers to data points around centroid[i]
    centroids = [None]*k                    # central points for clusters
    wcss = [None]*len(training_set)         # euclidean distances between data & centroids

    # Initially randomly select K observations as centroids
    for i in range(k):
        index_selected = np.random.randint(0, len(training_set))
        centroids[i]=training_set[index_selected]

    # Recalculate centroids and re-assign clusters until convergence.
    changed = True
    while changed:
        changed = False

        # Assign observations to the minimum distance centroid
        for i in range(len(training_set)):
            min_dist = sys.maxint
            min_index = sys.maxint

            for j in range(k):
                curr_dist = euclidean_distance(training_set[i], centroids[j])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_index = j

            wcss[min_index] = min_dist
            if training_set[i] not in clusters[min_index]:
                clusters[min_index].append(training_set[i])
                changed = True



        # If observations changed clusters, then recalculate the centroids.
        if changed is True:
            centroids = recalculate_centroids(clusters, centroids)

    name = 'k_%d_means' %k
    k_means_info = collections.namedtuple(name, ['centroids', 'clusters', 'wcss'])
    return k_means_info(centroids, clusters, wcss)


def euclidean_distance(observ1, observ2):
    result = 0.0
    for i in range(len(observ1) - 1):
        result += pow((observ1[i] - observ2[i]), 2)
    return math.sqrt(result)


def recalculate_centroids(clusters, centroids):
    new_centroids = [None]*len(centroids)

    # For each cluster, sum up the values of each feature. Then, take the average
    # of each each feature's sum. This will be the new centroid for the cluster.
    for i in range(len(clusters)):
        feature_averages = [0] * len(clusters[i])

        for observation in clusters[i]:
            for j in range(len(observation)):
                feature_averages[j] += observation[j]

        for k in range(len(feature_averages)):
            feature_averages[k] /= len(clusters[i])

        new_centroids[i] = feature_averages

    return new_centroids
