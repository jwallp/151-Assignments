import random as rand
import numpy as np
import matplotlib.pyplot as plt
import SampleWithoutReplacement as swr

TRAIN = 0
TEST = 1

'''Initialization'''
rand.seed(777)
means = list()
sd = list()
num_runs = [10,100,1000,10000,100000]

'''Prompt user to enter percent of data set to use for test set'''
threshold = int(input("Enter a threshold percentage: "))/100.0

'''Create a Sampler'''
sampler = swr.SampleWithoutReplacement('abalone.csv',threshold,TEST)

'''Run trials'''
for run in num_runs:
    for i in range(run):
        sampler.select()
    np_counts = np.asarray(sampler.get_counts())
    means.append(np.mean(np_counts)/float(run))
    sd.append(np.std(np_counts)/float(run))
    sampler.reset()
    print("Running",run, "trials ...")

'''Show results'''
print ("Mean: ",means)
print ("Standard Deviation: ",sd)

plt.plot(num_runs, means, 'ro')
plt.xlabel("# of Runs")
plt.xticks(num_runs, num_runs, rotation='vertical')
plt.margins(0.2)
plt.ylabel("Mean")
plt.show()
