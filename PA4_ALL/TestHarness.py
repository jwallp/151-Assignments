import random as rand
import numpy as np
import SampleWithoutReplacement as swr
import csv
import KMeans
import sys
from scipy.cluster.vq import vq, kmeans, whiten

'''Initialization'''
rand.seed(777)

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
print "Done adjusting abalone.csv ..."

sampler = swr.SampleWithoutReplacement('datasets/adjusted-abalone.csv', .10)
sampler.select()
sampler.z_scale()

training_set = sampler.get_training_set()
test_set = sampler.get_test_set()

global_wcss = list()


#blah = kmeans(np.array(training_set)[:, :-1], 1)

for i in [1, 2, 4, 8, 16]:
    print "For K=%d:" %i
    results = KMeans.k_means(training_set, i)
    print "->WCSS: %s" % results.wcss
    global_wcss.append(sum(results.wcss))
    print "->Centroids: %s" % results.centroids

    # Calculate Mean & SD
    cluster_mean = None
    cluster_SD = None

    weights_for_each_cluster = [None]*i
    for j in range(len(results.clusters)):
        print "Cluster %d :" % j
        arr = np.array(results.clusters[j])
        print arr
        cluster_mean = np.mean(arr[:, :-1], axis=0)
        cluster_SD = np.std(arr[:, :-1], axis=0)
        weights = np.linalg.lstsq(arr[:, :-1], arr[:, -1])[0]
        weights_for_each_cluster[j] = weights

        for k in range(len(cluster_mean)):
            print "      ->Feature %d mean = %f, SD = %f" % (k, cluster_mean[k], cluster_SD[k])

    # Cluster the test set
    test_clusters = [[] for a in range(i)]
    for j in range(len(test_set)):
        min_dist = sys.maxint
        min_index = sys.maxint

        for k in range(len(results.centroids)):
            curr_dist = KMeans.euclidean_distance(test_set[j], results.centroids[k])
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_index = k
        if test_set[j] not in test_clusters[min_index]:
            test_clusters[min_index].append(test_set[j])

    # Now predict y for the test clusters using the weights from training clusters
    cluster_predictions = [[] for a in range(i)]
    for index in range(i):
        current_weight = weights_for_each_cluster[index]
        for observation in test_clusters[index]:
            predicted_y = np.array(observation)[:-1].dot(np.array(current_weight))
            cluster_predictions[index].append(predicted_y)

    # Calculate RMSE by comparing predictions and actual values
    for index in range(i):
        for index2 in range(len(test_clusters[index])):
            current_cluster = test_clusters[index]
            actual = current_cluster[index2][-1]
            predicted = cluster_predictions[index][index2]
            difference = actual - predicted










