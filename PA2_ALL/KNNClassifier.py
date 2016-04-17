import sys
import KNNMath

class KNNClassifier:
    """
    This class runs K nearest neighbors given a test set and a training set.
    """

    def __init__(self, test_set, training_set):
        self.test_set = test_set
        self.training_set = training_set
        self.current_observation = None
        self.euc_dist_list = None

    def run(self, k_arr):
        count = 1
        all_classifications = list()

        count = 1
        for item in self.test_set:
            temp_list = list()
            for k in k_arr:
                temp_list.append(self.classify(item, k))
                sys.stdout.write("\r      -Classifying Observation %i" %count + "/%i for " %len(self.test_set) + "K=%i" %k)
                sys.stdout.flush()
            all_classifications.append(temp_list)
            count+=1
        print()
        return all_classifications

    def classify(self, observation, k):
        if self.current_observation is not observation:
            'Find euclidean distance to all observations in training set'
            self.euc_dist_list = list()
            for i in range(len(self.training_set)):
                self.euc_dist_list.append((KNNMath.euclidean_distance(observation, self.training_set[i]), i))

            'Sort list of (euclidean distance, observation#). Will automatically sort by distance first'
            self.euc_dist_list.sort()
        else:
            self.current_observation = observation

        'Take first k entries of the list and use it to categorize the current observation'
        first_k_entries = list()
        for i in range(k):
            observation_num = self.euc_dist_list[i][1]
            first_k_entries.append(self.training_set[observation_num][len(self.training_set[0])-1])

        return KNNMath.mode(first_k_entries)

