################################################################################
PA2 for CSE 151
Names: Jessica Ng, Jaskanwal Pawar
PID: A10683076, A97059386
################################################################################
Important Notes-----------------------------------------------------------------:
Because the programming assignment asks us to perform KNN classification on ALL
K={1,3,5,7,9}, we chose to also save the confusion matrices produced by the
classification rather than only the ones for the K that produces the smallest
error rate.

KNN Classification was not optimized for runs on the SAME datasets but MULTIPLE K.
Runtime could have been saved by saving the euclidean distance list for an
observation, and then running that for all K={1,3,5,7,9} before moving onto the
next observation. Instead KNNClassifier works by classifying an entire dataset for
one K.

To Run------------------------------------------------------------------------:
Move to the directory PA2_ALL.
Type the following into the console.
    python TestHarness

Results-------------------------------------------------------------------------:
The K that produces the least error rate is K = 9 among the possible choices
{1,3,5,7,9}.

The confusion matrices for K=9 may be found in 'PA2_ALL/confusion_matrices/K9'

Dependencies--------------------------------------------------------------------:
The following are required for the program to execute
- Python 3
- numPy

In order for this program to work, all data files must be in 'PA2_ALL/datasets'
and the names of the file must not be changed.