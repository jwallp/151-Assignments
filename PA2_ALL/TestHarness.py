import random as rand
import numpy as np
import matplotlib.pyplot as plt
import SampleWithoutReplacement as swr
import KNNClassifier as knn
import ExemplarProvider as ep
import sys
import csv
import KNNMath

TRAIN = 0
TEST = 1

'''Initialization'''
rand.seed(777)

'''Prompt user to enter percent of data set to use for test set'''
threshold = int(input("Enter a test set percentage: "))/100.0

'''Adjust abalone.csv'''
abalone_file = open('datasets/abalone.csv')
reader = csv.reader(abalone_file)

#new format:[M, F, I, rest of stuff]
with open('datasets/adjusted-abalone.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    for row in reader:
        sex = row[0]
        new_row = list()

        new_row.append(1 if sex == 'M' else 0)
        new_row.append(1 if sex == 'F' else 0)
        new_row.append(1 if sex == 'I' else 0)

        for i in range(1, len(row)):
            new_row.append(row[i])

        csv_writer.writerow(new_row)
abalone_file.close()
print("Done adjusting abalone.csv ...")

data_files = ['adjusted-abalone', '3percent-miscategorization',
              '10percent-miscategorization', 'Seperable']
for data_file in data_files:
    '''Create a Sampler'''
    file_name = 'datasets/%s.csv' %(data_file)
    print(file_name, "is being run:")
    sampler = swr.SampleWithoutReplacement(file_name, threshold)
    print("   ->Selecting training and test sets ... ")
    sampler.select()
    print("   ->Normalizing training and test sets ...")
    sampler.z_scale()

    test_set = sampler.get_test_set()
    training_set = sampler.get_training_set()
    classifier = knn.KNNClassifier(test_set, training_set)

    k_arr = [1,3,5,7,9]
    for k in k_arr:
        print("   ->Running KNN for", data_file, "on K =", k,":")
        classified = classifier.run(k)
        total = len(classified)

        miss = 0
        """
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        """

        name = 'confusion_matrices/K%d/%s-K%d-matrix.csv'%(k, data_file, k)
        write_file = open(name, 'w', newline='')
        csv_writer = csv.writer(write_file, delimiter=',')

        max_val = KNNMath.max_of_lists([test_set, training_set])
        min_val = KNNMath.min_of_lists([test_set, training_set])
        matrix = [[0 for i in range(min_val, max_val+1)] for j in range(min_val, max_val+1)]

        for i in range(total):
            actual = int(test_set[i][len(test_set[0])-1])
            prediction = int(classified[i])

            'if category = (1,25) then category 24 corresponds to column and row (24-1)=23'
            'if category = (0,1) then category 1 corresponds to column and row (1-0)=1'
            matrix[actual-min_val][prediction-min_val] += 1

            if prediction != actual:
                miss+=1

        for row in matrix:
            csv_writer.writerow(row)

        accuracy = 'Error rate: %f' %(miss/float(total)*100)
        write_file.write(accuracy)

        write_file.close()
        print("      -%s"%accuracy)
