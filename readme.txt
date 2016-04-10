################################################################################
PA1 for CSE 151
Names: Jessica Ng,
PID: A10683076,
################################################################################

To Run------------------------------------------------------------------------:
For the purpose of this assignment, we created 2 different versions.

In folder PA1, you will find a functional implementation of the assignment.
We do not use any objects. pa1.py contains methods to sample the csv file
and test.py contains code that will run several tests on the data set that's
opened.
To run this version, go to the folder PA1 and type:
    python test
You will then be prompted to type in a percentage where 10 = 10%, 20 = 20%, etc.

In folder PA1OOP, you will find an OOP implementation of the assignment.
We follow the format given by the professor's slides with an ExemplarProvider,
SampleWithoutReplacement, and TestHarness class. ExemplarProvider opens and
handles the file, SampleWithoutReplacement handles sampling of the data, and
TestHarness utilizes SampleWithoutReplacement to sample the data set.
To run this version, go to the folder PA1OOP and type:
    python TestHarness
You will then be prompted to type in a percentage where 10 = 10%, 20 = 20%, etc.

Results-------------------------------------------------------------------------:
You can find a graph of # of runs Vs Means in graphMeanxRuns.png or by running the
program.

You can find the resulting Means and SD from one run of the program in Results.txt

Dependencies--------------------------------------------------------------------:
The following are required for the program to execute
- Python 3
- numPy
- matplotlib (pyplot)

abalone.csv must be in the folder PA1/PA1OOP to run.

