import collections
import numpy as np
import math
import sys


def k_means(training_set, k):
    """
    :param training_set: list of observations
    :param k: number of clusters to create
    :return: tuple of centroids, clusters, WCSS
    """

    clusters = [[] for i in range(k)]       # cluster[i] refers to data points around centroid[i]
    clusters_indices = [[] for i in range(k)]
    centroids = [None]*k                    # central points for clusters
    wcss = [None]*len(training_set)         # euclidean distances between data & centroids

    # Initially randomly select K observations as centroids
    indices_selected = list()
    for i in range(k):
        while True:
            index_selected = np.random.randint(0, len(training_set))
            if index_selected not in indices_selected:
                centroids[i]=training_set[index_selected]
                indices_selected.append(index_selected)
                break

    # Recalculate centroids and re-assign clusters until convergence.
    changed = True
    while changed:
        changed = False
        clusters = [[] for i in range(k)]
        clusters_indices = [[] for i in range(k)]

        # Assign observations to the minimum distance centroid
        for i in range(len(training_set)):
            min_dist = sys.maxint
            min_index = sys.maxint

            for j in range(k):
                curr_dist = euclidean_distance(training_set[i], centroids[j])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_index = j

            wcss[i] = min_dist
            clusters[min_index].append(training_set[i])
            clusters_indices[min_index].append(i)

        # If observations changed clusters, then recalculate the centroids.
        new_centroids = recalculate_centroids(clusters, centroids)
        for centroid in new_centroids:
            if centroid not in centroids:
                changed = True
        centroids = new_centroids

    name = 'k_%d_means' %k
    k_means_info = collections.namedtuple(name, ['centroids', 'clusters', 'indices', 'wcss'])
    return k_means_info(centroids, clusters, clusters_indices, wcss)


def k_means2(training_set, k, start_centroids):
    """
    :param training_set: list of observations
    :param k: number of clusters to create
    :return: tuple of centroids, clusters, WCSS
    """

    clusters = [[] for i in range(k)]               # cluster[i] refers to data points around centroid[i]
    clusters_indices = [[] for i in range(k)]
    centroids = start_centroids                    # central points for clusters
    wcss = [None]*len(training_set)         # euclidean distances between data & centroids

    # Recalculate centroids and re-assign clusters until convergence.
    changed = True
    while changed:
        changed = False
        clusters = [[] for i in range(k)]
        clusters_indices = [[] for i in range(k)]

        # Assign observations to the minimum distance centroid
        for i in range(len(training_set)):
            min_dist = sys.maxint
            min_index = sys.maxint

            for j in range(k):
                curr_dist = euclidean_distance(training_set[i], centroids[j])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_index = j

            wcss[i] = min_dist
            clusters[min_index].append(training_set[i])
            clusters_indices[min_index].append(i)

        # If observations changed clusters, then recalculate the centroids.
        new_centroids = recalculate_centroids(clusters, centroids)
        for centroid in new_centroids:
            if centroid not in centroids:
                changed = True
        centroids = new_centroids

    name = 'k_%d_means' %k
    k_means_info = collections.namedtuple(name, ['centroids', 'clusters', 'indices', 'wcss'])
    return k_means_info(centroids, clusters, clusters_indices, wcss)


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
        feature_averages = [0.0] * len(clusters[i][0])

        for observation in clusters[i]:
            for j in range(len(observation)):
                feature_averages[j] += observation[j]

        for k in range(len(feature_averages)):
            feature_averages[k] /= len(clusters[i])

        new_centroids[i] = feature_averages

    return new_centroids
