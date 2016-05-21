import random as rand
import numpy as np
import SampleWithoutReplacement as swr
import csv

'''Initialization'''
rand.seed(777)

'''Prompt user to enter percent of data set to use for test set'''
threshold = int(input("Enter a test set percentage: "))/100.0

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
print("Done adjusting abalone.csv ...")




