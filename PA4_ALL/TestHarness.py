import random as rand
import numpy as np
import SampleWithoutReplacement as swr
import csv
import KMeans
import sys
import math
import collections
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt


def adjust_abalone_dataset():
    """
    Rewrite the abalone dataset so that it does not contain non-numeric datapoints

    :return: N/A
    """
    '''Adjust abalone.csv'''
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


def calculate_cluster_values(cluster):
    """
    Calculates the mean, SD, and weights of the given cluster.

    :param cluster: The clusters that test set observations have been assigned to. Has format
                    [list of [clusters of[observations of n features]]]
    :return: A tuple which contains the mean and SD of each feature in the cluster. It also
            contains the dimensional weights of x in Ax=b. Has format
                    Tuple(mean, sd, weights)
    """
    arr = np.array(cluster)
    cluster_mean = np.mean(arr[:, :-1], axis=0)
    cluster_SD = np.std(arr[:, :-1], axis=0)
    weights = np.linalg.lstsq(arr[:, :-1], arr[:, -1])[0]

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

    for i in range(len(input_set)):
        min_dist = sys.maxint
        min_index = sys.maxint

        for j in range(len(centroids)):
            curr_dist = KMeans.euclidean_distance(input_set[i], centroids[j])
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_index = j
        # TODO: not in np.array changes output
        if input_set[i] not in cluster_set[min_index]:
        # if input_set[i] not in np.array(cluster_set[min_index]):
            cluster_set[min_index].append(input_set[i])

    return cluster_set


def predict_categories(clusters, weights):
    """
    Predicts the categories of each observation in the clusters using the weights given.

    :param clusters: K clusters of observations with format
                    [list of [clusters of [observations of n features]]]
    :param weights: Dimensional weights from solving x for Ax=b with least squares. Has format of
                    [list of [clusters of weights]] from solving x for Ax=b with least squares
    :return: A list of predicted categories for each observation in the clusters. Has format of
                    [list of [clusters of predicted categories]]
    """
    predictions = [[] for a in range(len(clusters))]
    for index in range(len(clusters)):
        current_weight = weights[index]
        for observation in clusters[index]:
            predicted_y = np.array(observation)[:-1].dot(np.array(current_weight))
            predictions[index].append(predicted_y)
    return predictions


def find_RMSE(input_set, clusters, predictions):
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
    for cluster in range(len(clusters)):
        current_cluster = clusters[cluster]
        for observation in range(len(test_clusters[cluster])):
            actual_category = current_cluster[observation][-1]
            predicted_category = predictions[cluster][observation]
            sum_of_differences_squared += pow(actual_category - predicted_category, 2)
    return math.sqrt(sum_of_differences_squared / len(input_set))


# TODO:
def clean_columns(clusters):
    for cluster in clusters:
        for i in range(len(cluster[0])):
        # for i in range(cluster.shape[1]):
            if np.allclose(np.array(cluster)[:, i], np.zeros(len(cluster))):
                # TODO: actually delete column from the input object reference
                cluster = np.delete(np.array(cluster), i, axis=1).tolist()

    return clusters


if __name__ == "__main__":
    # Initialization
    rand.seed(777)
    adjust_abalone_dataset()

    # Random sample the dataset and then scale it.
    sampler = swr.SampleWithoutReplacement('datasets/adjusted-abalone.csv', .10)
    sampler.z_scale()
    training_set = sampler.get_training_set()
    test_set = sampler.get_test_set()

    # reintroduce bias column to correct RMSE values
    training_set = np.insert(training_set, -1, 1, axis=1).tolist()
    test_set = np.insert(test_set, -1, 1, axis=1).tolist()

    global_wcss = list()
    global_rmse = list()

    # Run K-means on the data set and output results from it
    for i in [1, 2, 4, 8, 16]:

        # Run K-means on the training set and store the data
        results = KMeans.k_means(training_set, i)
        global_wcss.append(sum(results.wcss))
        # clean_columns(results.clusters)

        # Calculate the mean, sd, and weights of all clusters
        cluster_weights = [None]*i
        cluster_info = list()
        for j in range(len(results.clusters)):
            info = calculate_cluster_values(results.clusters[j])
            cluster_info.append(info)
            cluster_weights[j]=list(info.weights)

        # assign all observations in the test set to clusters
        test_clusters = assign_to_cluster(test_set, results.centroids)
        # test_clusters = clean_columns(test_clusters)

        # Now predict y for the test clusters using the weights from training clusters
        cluster_predictions = predict_categories(test_clusters, cluster_weights)

        # Calculate RMSE by comparing predictions and actual values
        rmse = find_RMSE(test_set, test_clusters, cluster_predictions)
        print "RMSE for k=%d: %f" % (i, rmse)

        global_rmse.append(rmse)

        # Write results to the console.
        print "For K=%d:" % i
        print "->Centroids:"
        for j in range(len(results.centroids)):
            print "\tCentroid %d:\n\t%s\n" % (j, results.centroids[j])
        for j in range(len(cluster_info)):
            print "\tCluster %d:" % j
            current_info = cluster_info[j]
            for k in range(len(current_info.mean)):
                print"\t\t Feature %d: mean: %f  sd: %f" % (k, current_info.mean[k], current_info.sd[k])
    print "WCSS sums for all K: %s" % global_wcss
    print "RMSE for all K: %s" % global_rmse
    # print "Sum of all RMSE's: %f" % sum(global_rmse)

    # Graph WCSS vs K and graph RMSE vs K.
    # Their values can be found in global_wcs and global_rmse
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('plots.pdf')

    plt.plot([1, 2, 4, 8, 16], global_rmse)
    plt.xlabel("k")
    plt.ylabel("RMSE")
    plt.title("RMSE vs K")
    plt.savefig(pp, format='pdf')

    # clear first plot
    plt.cla()
    plt.clf()
    plt.close()

    plt.plot([1, 2, 4, 8, 16], global_wcss)
    plt.xlabel("k")
    plt.ylabel("WCSS")
    plt.title("WCSS vs K")
    plt.savefig(pp, format='pdf')

    pp.close()

