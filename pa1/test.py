import pa1
import random as rand
import numpy as np


threshold = int(input("Enter a threshold percentage: "))/100.0
counts = None
rand.seed(777)

means = list()
sd = list()

for i in range(10):
    counts = pa1.read_user_input(threshold, counts)
np_counts = np.asarray(counts)
means.append(np.mean(np_counts))
sd.append(np.std(np_counts))
print "10"

rand.seed(777)
counts = None
for i in range(100):
    counts = pa1.read_user_input(threshold, counts)
np_counts = np.asarray(counts)
means.append(np.mean(np_counts))
sd.append(np.std(np_counts))
print "100"

rand.seed(777)
counts = None
for i in range(1000):
    counts = pa1.read_user_input(threshold, counts)
np_counts = np.asarray(counts)
means.append(np.mean(np_counts))
sd.append(np.std(np_counts))
print "1,000"

rand.seed(777)
counts = None
for i in range(10000):
    counts = pa1.read_user_input(threshold, counts)
np_counts = np.asarray(counts)
means.append(np.mean(np_counts))
sd.append(np.std(np_counts))
print "10,000"

rand.seed(777)
counts = None
for i in range(100000):
    counts = pa1.read_user_input(threshold, counts)
np_counts = np.asarray(counts)
means.append(np.mean(np_counts))
sd.append(np.std(np_counts))
print "done"

print means
print sd
