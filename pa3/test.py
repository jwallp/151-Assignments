import numpy as np
import Householder as hh
import csv
import SampleWithoutReplacement as swr
import random as rand

np.set_printoptions(threshold=np.nan)


'''Initialization'''
rand.seed(777)

'''Prompt user to enter percent of data set to use for test set'''
threshold = int(input("Enter a test set percentage: "))/100.0

A = np.mat([[0.8147, 0.0975, 0.1576], [0.9058, 0.2785, 0.9706],
                 [0.1270, 0.5469, 0.9572], [0.9134, 0.9575, 0.4854],
                 [0.6324, 0.9649, 0.8003]])
B = np.mat([
    [12., -51., 4.],
    [6., 167., -68.],
    [-4., 24., -41.]
])

'''Adjust abalone.csv'''
abalone_file = open('datasets/abalone.csv')
reader = csv.reader(abalone_file)

#new format:[M, F, I, rest of stuff]
with open('datasets/adjusted-abalone.csv', 'w') as csvfile:
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
print("Done adjusting abalone.csv ...")

'''Create a Sampler'''
file_name = 'datasets/adjusted-abalone.csv'
print(file_name, "is being run:")
sampler = swr.SampleWithoutReplacement(file_name, threshold)
print("   ->Selecting training and test sets ... ")
sampler.select()

a = np.matrix(sampler.get_training_set())
c = np.mat([
    [1.0,-1.0,-1.0],
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 1.0],
    [2.0,-2.0, 1.0],
    [3.0, 2.0, 1.0]
])
print c
householder = hh.Householder(c)
print householder.get_R()
#print householder.back_solve()
b = np.matrix(sampler.get_training_set())
#test1 = householder.regression_prediction(b)
#test2 = householder.regression_predictionb(b)


