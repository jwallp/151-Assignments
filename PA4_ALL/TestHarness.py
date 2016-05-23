import random as rand
import numpy as np
import SampleWithoutReplacement as swr
import csv
import KMeans
import sys
import math
import collections
import Householder as hh
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


def adjust_abalone_dataset():
    """
    Rewrite the abalone dataset so that it does not contain non-numeric datapoints

    :return: N/A
    """
    abalone_file = open('datasets/abalone.csv')
    reader = csv.reader(abalone_file)

    # new format:[M, F, I, rest of stuff]
    with open('datasets/adjusted-abalone.csv', 'wb') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        for row in reader:
            sex = row[0]
            new_row = list()

            new_row.append(1.0 if sex == 'M' else 0.0)
            new_row.append(1.0 if sex == 'F' else 0.0)
            new_row.append(1.0 if sex == 'I' else 0.0)

            for i in range(1, len(row)):
                new_row.append(row[i])

            csv_writer.writerow(new_row)
    abalone_file.close()
    # print "Done adjusting abalone.csv ..."


def calculate_cluster_values(results, j, unscaled_set):
    """
    Calculates the mean, SD, and weights of the given cluster.

    :param cluster: The clusters that test set observations have been assigned to. Has format
                    [list of [clusters of[observations of n features]]]
    :return: A tuple which contains the mean and SD of each feature in the cluster. It also
            contains the dimensional weights of x in Ax=b. Has format
                    Tuple(mean, sd, weights)
    """

    cluster = results.clusters[j]
    arr = np.array(cluster)

    cluster_indices = results.indices[j]
    weights_matrix = list()
    for index in cluster_indices:
        weights_matrix.append(unscaled_set[index])
    weights_matrix = np.array(weights_matrix)

    np.savetxt("test.csv", arr, delimiter=",")

    cluster_mean = np.mean(arr[:, :-1], axis=0)
    cluster_SD = np.std(arr[:, :-1], axis=0)
    weights = np.linalg.lstsq(weights_matrix[:, :-1], weights_matrix[:, -1])[0]

    cluster_info = collections.namedtuple("clusterInfo", ['mean', 'sd', 'weights'])
    return cluster_info(cluster_mean, cluster_SD, weights)


def assign_to_cluster(input_set, centroids):
    """
    Assigns every observation in the input_set to a cluster. Clusters are centered around the
    given centroids

    :param input_set: List of observations from the test set to put into a cluster. Has format
                    [list of [observations]]
    :param centroids: List of centroids from running K-means on a training set. Has format
                    [list of centroids]
    :return: The clusters that test set observations have been assigned to. Has format
                    [list of [clusters of[observations of n features]]]
    """
    if len(centroids) < 1:
        raise Exception('No centroids were given.')
    if len(input_set) < 1:
        raise Exception('No input observations were given.')

    cluster_set = [[] for a in range(len(centroids))]
    cluster_indices = [[] for a in range(len(centroids))]

    for i in range(len(input_set)):
        min_dist = sys.maxint
        min_index = sys.maxint

        for j in range(len(centroids)):
            curr_dist = KMeans.euclidean_distance(input_set[i], centroids[j])
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_index = j
        if input_set[i] not in cluster_set[min_index]:
            cluster_set[min_index].append(input_set[i])
            cluster_indices[min_index].append(i)

    assignment = collections.namedtuple("clusterAssigment", ['clusters', 'indices'])
    return assignment(cluster_set, cluster_indices)


def predict_categories(clusters, weights, unscaled_test_set):
    """
    Predicts the categories of each observation in the clusters using the weights given.

    :param clusters: K clusters of observations with format
                    [list of [clusters of [observations of n features]]]
    :param weights: Dimensional weights from solving x for Ax=b with least squares. Has format of
                    [list of [clusters of weights]] from solving x for Ax=b with least squares
    :return: A list of predicted categories for each observation in the clusters. Has format of
                    [list of [clusters of predicted categories]]
    """
    predictions = [[] for a in range(len(clusters.indices))]
    for index in range(len(clusters.indices)):
        current_weight = weights[index]
        for observation_index in clusters.indices[index]:
            predicted_y = np.array(unscaled_test_set[observation_index])[:-1].dot(np.array(current_weight))
            predictions[index].append(predicted_y)
    return predictions


def find_RMSE(unscaled_set, clusters, predictions):
    """
    Calculates the RMSE by looking at the difference between the actual category (given by clusters)
    and the predicted category (given by predictions).

    :param input_set: List of observations from the test set
                    [list of [observations of n features]]
    :param clusters: K clusters of observations with format
                    [list of [clusters of [observations of n features]]]
    :param predictions: A list of predicted categories for each observation in the clusters. Has format of
                    [list of [clusters of predicted categories]]
    :return: float value representing the RMSE
    """
    sum_of_differences_squared = 0
    for cluster in range(len(clusters.indices)):
        current_cluster_indices = clusters.indices[cluster]
        for j in range(len(current_cluster_indices)):
            actual_category = unscaled_set[current_cluster_indices[j]][-1]
            predicted_category = predictions[cluster][j]
            sum_of_differences_squared += pow(actual_category - predicted_category, 2)
    return math.sqrt(sum_of_differences_squared / len(unscaled_set))


if __name__ == "__main__":
    # Initialization
    adjust_abalone_dataset()

    # Random sample the dataset and then scale it.
    rand.seed(777)
    sampler = swr.SampleWithoutReplacement('datasets/adjusted-abalone.csv', .10)
    sampler.z_scale()
    training_set = sampler.get_training_set()
    test_set = sampler.get_test_set()

    # Tried to see if it made a difference if QR was performed on unscaled datasets
    rand.seed(777)
    sampler2 = swr.SampleWithoutReplacement('datasets/adjusted-abalone.csv', .10)
    unscaled_training_set = sampler2.get_training_set()
    unscaled_test_set = sampler2.get_test_set()

    global_wcss = list()
    global_rmse = list()

    # Run K-means on the data set and output results from it
    for i in [1, 2, 4, 8, 16]:

        # Run K-means on the training set and store the data
        results = KMeans.k_means(training_set, i)
        global_wcss.append(sum(results.wcss))

        # Calculate the mean, sd, and weights of all clusters
        cluster_weights = [None]*i
        cluster_info = list()
        for j in range(len(results.clusters)):
            info = calculate_cluster_values(results, j, unscaled_training_set)
            cluster_info.append(info)
            cluster_weights[j]=list(info.weights)

        # assign all observations in the test set to clusters
        test_clusters = assign_to_cluster(test_set, results.centroids)

        # Now predict y for the test clusters using the weights from training clusters
        cluster_predictions = predict_categories(test_clusters, cluster_weights, unscaled_test_set)

        # Calculate RMSE by comparing predictions and actual values
        rmse = find_RMSE(unscaled_test_set, test_clusters, cluster_predictions)

        global_rmse.append(rmse)

        # Write results to the console.
        print "For K=%d:" % i
        print "->Centroids:"
        for j in range(len(results.centroids)):
            print "\tCentroid %d:\n\t%s\n" % (j, results.centroids[j])
        for j in range(len(cluster_info)):
            print "\tCluster %d :" % j
            current_info = cluster_info[j]
            for k in range(len(current_info.mean)):
                print"\t\t Feature %d: mean:%f  sd:%f" % (k, current_info.mean[k], current_info.sd[k])
    print "WCSS sums for all K: %s" % global_wcss
    print "RMSE for all K: %s" % global_rmse

    # TODO: Graph WCSS vs. K and graph RMSE vs. K. Their values can be found in global_wcs and global_rmse