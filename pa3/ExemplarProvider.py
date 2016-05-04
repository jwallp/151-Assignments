import csv
import numpy as np

ERROR_COLUMN = 1
TEST_SET = 0
TRAINING_SET = 1


class ExemplarProvider:
    """
    This class provides and holds data for the sampler. It opens up the data
    set file, then deals with any necessary transformations to the data set.
    """

    def __init__(self, filename):
        self.file = open(filename)
        self.reader = csv.reader(self.file)

        'Convert our data set into a list of list of floats'
        self.test_set = list()
        self.training_set = list()

        'Retain the length of our list'
        self.N = len(list(self.reader))
        self.file.seek(0)

    def read_next_line(self):
        return next(self.reader)

    'Normalize entire data set except for the last column'
    def z_scale(self):
        for i in range(len(self.test_set[0])- ERROR_COLUMN):
            self.z_scale_column(i)

    'Normalize a column'
    def z_scale_column(self, column):
        items = list()

        for subList in self.test_set:
            items.append(subList[column])
        for subList in self.training_set:
            items.append(subList[column])

        items_arr = np.asarray(items)
        col_mean = np.mean(items_arr)
        col_sd = np.std(items_arr)

        #print("mean: ", col_mean, "sd: ", col_sd)

        for subList in self.test_set:
            subList[column] = (subList[column] - col_mean)/col_sd
        for subList in self.training_set:
            subList[column] = (subList[column] - col_mean)/col_sd

    'Adds the next row in the csv file to a specified set'
    def add_to_set(self, type):
        row = list()
        for item in next(self.reader):
            row.append(float(item))

        if type == TEST_SET:
            self.test_set.append(row)
        elif type == TRAINING_SET:
            self.training_set.append(row)

    'Getters & Setters'
    def get_file_size(self):
        return self.N

    def get_training_set(self):
        return self.training_set

    def get_test_set(self):
        return self.test_set