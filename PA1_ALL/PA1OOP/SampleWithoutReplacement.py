import ExemplarProvider as ep
import random as rand


class SampleWithoutReplacement:

    def __init__(self, filename, sample_percent, flag):
        self.filename = filename
        self.sample_percent = sample_percent
        self.exemplar_provider = ep.ExemplarProvider(filename)

        self.total = self.get_file_size()
        self.Nr_original = self.total
        self.Nn_original = round(self.sample_percent * self.Nr_original)
        self.counts = [0] * self.total

    def select(self):
        Nn = self.Nn_original
        Nr = self.Nr_original

        while Nn > 0:
            draw = rand.random()
            if draw < Nn/Nr:
                #self.exemplar_provider.read_next_line()
                self.counts[self.total - Nr] += 1
                Nn -= 1
            Nr -= 1
        return self.counts

    def reset(self):
        for i in range(self.total):
            self.counts[i]=0

    def get_counts(self):
        return self.counts

    def get_file_size(self):
        return self.exemplar_provider.get_file_size()


