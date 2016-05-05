import os
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import SampleWithoutReplacement as swr
import ExemplarProvider as ep
import Householder as hh
import csv

TRAIN = 0
TEST = 1

'''Initialization'''
rand.seed(777)

'''Prompt user to enter percent of data set to use for test set'''
threshold = int(input("Enter a test set percentage: "))/100.0

'''Adjust abalone.csv'''
if not os.path.isfile('datasets/adjusted-abalone.csv'):
    abalone_file = open('datasets/abalone.csv')
    reader = csv.reader(abalone_file)

    #new format:[M, F, I, rest of stuff]
    with open('datasets/adjusted-abalone.csv', 'w') as csvfile:
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

data_files = ['regression-0.05', 'regression-A', 'regression-B',
              'regression-C', 'adjusted-abalone']

for data_file in data_files:
    '''Create a Sampler'''
    file_name = 'datasets/%s.csv' %(data_file)
    print("\n%s is being run:") % (data_file)

    sampler = swr.SampleWithoutReplacement(file_name, threshold)
    print("\t->Selecting training and test sets ... ")
    sampler.select()
    print("\t->Normalizing training and test sets ...")
    # TODO: keep z-scale?
    # sampler.z_scale()

    training_set = np.mat(sampler.get_training_set())
    test_set = np.mat(sampler.get_test_set())

    '''
    Perform QR decomposition of training set in order to solve least
    squares regresion
    '''
    trainer = hh.Householder(training_set)
    trainer.get_R()

    predictions = trainer.regression_prediction(test_set)
    actual = test_set[:, -1].T
    difference = predictions - actual

    ''' Compute RMSE of predictions for current dataset '''
    rmse = np.sqrt(difference.dot(difference.T)[0, 0] / test_set.shape[0])
    print "\t->RMSE=%s" % rmse

