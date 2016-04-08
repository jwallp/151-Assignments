import pa1
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
num_runs = [10]

'''Run through algorithm num_runs[i]'''
for run in num_runs:
    for i in range(run):
        counts = pa1.read_user_input(threshold, datalist, counts, N)
        print (counts)
    np_counts = np.asarray(counts)
    means.append(np.mean(np_counts)/float(run))
    sd.append(np.std(np_counts)/float(run))
    print (run)
    counts = None

"""
# rand.seed(777)
counts = None
for i in range(100):
    counts = pa1.read_user_input(threshold, datalist, counts, N)
np_counts = np.asarray(counts)
means.append(np.mean(np_counts)/100.0)
sd.append(np.std(np_counts)/100.0)
print ("100")

# rand.seed(777)
counts = None
for i in range(1000):
    counts = pa1.read_user_input(threshold, datalist, counts, N)
np_counts = np.asarray(counts)
means.append(np.mean(np_counts)/1000.0)
sd.append(np.std(np_counts)/1000.0)
print ("1,000")


# rand.seed(777)
counts = None
for i in range(10000):
    counts = pa1.read_user_input(threshold, datalist, counts, N)
np_counts = np.asarray(counts)
means.append(np.mean(np_counts)/10000.0)
sd.append(np.std(np_counts)/10000.0)
print ("10,000")

rand.seed(777)
counts = None
for i in range(100000):
    counts = pa1.read_user_input(threshold, datalist, counts, N)
np_counts = np.asarray(counts)
means.append(np.mean(np_counts)/100000.0)
sd.append(np.std(np_counts)/100000.0)
print "done"
"""

print (means)
print (sd)
# for i in range(len(means)):
#    means[i] = np.log10(means[i])

# num_runs = [10,100,1000,10000]
# num_runs = [1,2,3,4]
plt.plot(num_runs, means)
# plt.ylim([0.1,0.1001])
plt.xlabel("# of Runs")
plt.ylabel("Mean")
plt.show()

