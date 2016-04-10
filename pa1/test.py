import PA1
import random as rand
import numpy as np
import csv
import matplotlib.pyplot as plt


'''Open the data set file'''
N = None
datalist = None
with open('abalone.csv') as file:
    reader = csv.reader(file)
    datalist = list(reader)
    N = len(datalist)

'''Prompt user to enter percent of data set to use for test set'''
threshold = int(input("Enter a threshold percentage: "))/100.0

'''Initialize random generator and other info'''
rand.seed(777)

counts = None
means = list()
sd = list()
num_runs = [10,100,1000,10000,100000]

'''Run through algorithm num_runs[i]'''
for run in num_runs:
    for i in range(run):
        counts = PA1.read_user_input(threshold, datalist, counts, N)
        print (counts)
    np_counts = np.asarray(counts)
    means.append(np.mean(np_counts)/float(run))
    sd.append(np.std(np_counts)/float(run))
    print (run)
    counts = None

print (means)
print (sd)

plt.plot(num_runs, means, 'ro')
plt.xlabel("# of Runs")
plt.xticks(num_runs, num_runs, rotation='vertical')
plt.margins(0.2)
plt.ylabel("Mean")
plt.show()

