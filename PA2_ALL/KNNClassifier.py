import sys
import Math

class KNNClassifier:
    """
    This class runs K nearest neighbors given a test set and a training set.
    """

    def __init__(self, test_set, training_set):
        self.test_set = test_set
        self.training_set = training_set

    def run(self, k):
        classifications = list()
        count = 1
        for item in self.test_set:
            classifications.append(self.classify(item, k))
            sys.stdout.write("\r      -Classifying %i" %count + "/%i" %len(self.test_set))
            count+=1
            sys.stdout.flush()
        print()
        return classifications

    def classify(self, observation, k):
        'Find euclidean distance to all observations in training set'
        euc_dist_list = list()
        for i in range(len(self.training_set)):
            euc_dist_list.append((Math.euclidean_distance(observation, self.training_set[i]), i))

        'Sort list of (euclidean distance, observation#). Will automatically sort by distance first'
        euc_dist_list.sort()

        'Take first k entries of the list and use it to categorize the current observation'
        first_k_entries = list()
        for i in range(k):
            observation_num = euc_dist_list[i][1]
            first_k_entries.append(self.training_set[observation_num][len(self.training_set[0])-1])

        return Math.mode(first_k_entries)

