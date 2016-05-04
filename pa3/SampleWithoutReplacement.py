import ExemplarProvider as ep
import random as rand


class SampleWithoutReplacement:
    """
    This class samples a data set. It splits the data set by either placing a
    data point in the training set or test set.
    """

    def __init__(self, filename, sample_percent):
        self.exemplar_provider = ep.ExemplarProvider(filename)

        self.total = self.get_file_size()
        self.Nr_original = self.total
        self.Nn_original = round(sample_percent * self.Nr_original)
        self.counts = [0] * self.total

    def select(self):
        Nn = self.Nn_original
        Nr = self.Nr_original

        for i in range(self.total):
            draw = rand.random()
            if draw < Nn/Nr:
                self.exemplar_provider.add_to_set(ep.TEST_SET)
                self.counts[self.total - Nr] += 1
                Nn -= 1
            else:
                self.exemplar_provider.add_to_set(ep.TRAINING_SET)
            Nr -= 1

        return self.counts

    def reset(self):
        for i in range(self.total):
            self.counts[i]=0

    def z_scale(self):
        self.exemplar_provider.z_scale()

    def get_counts(self):
        return self.counts

    def get_file_size(self):
        return self.exemplar_provider.get_file_size()

    def get_training_set(self):
        return self.exemplar_provider.get_training_set()

    def get_test_set(self):
        return self.exemplar_provider.get_test_set()


