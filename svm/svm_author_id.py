#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# flag to reduce the training set to 1%
# reducedTrainingSet = 1

"""
Docs links:
http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
http://scikit-learn.org/stable/modules/svm.html#using-python-functions-as-kernels
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

"""

#########################################################
### your code goes here ###


def predict(reducedTrainingSet, kernel, C, features_train, features_test, labels_train):
    from sklearn.svm import SVC

    if reducedTrainingSet:
        features_train = features_train[:len(features_train)/100]
        labels_train = labels_train[:len(labels_train)/100]
        print "** Using reduced training set! **"

    print "C = ", C, " kernel = ", kernel

    classifier = SVC(kernel=kernel, C=C)

    t0 = time()
    classifier.fit(features_train, labels_train)
    print "training time (fit):", round(time()-t0, 3), "s"

    t0 = time()
    labels_predicted = classifier.predict(features_test)
    print "predict time:", round(time()-t0, 3), "s"

    return labels_predicted

def calc_accuracy(reducedTrainingSet, kernel, C, features_train, features_test, labels_train, labels_test):
    from sklearn.metrics import accuracy_score

    labels_predicted = predict(reducedTrainingSet, kernel, C, features_train, features_test, labels_train)

    t0 = time()
    accuracy = accuracy_score(labels_test, labels_predicted)
    print "accuracy_score time:", round(time()-t0, 3), "s"

    print "accuracy: ", accuracy

    return accuracy

## parameters
#kernel = 'rbf'
#reducedTrainingSet = 1

#calc_accuracy(reducedTrainingSet, kernel, 10**4, features_train, features_test, labels_train, labels_test)

labels_predicted = predict(0, 'rbf', 10**4, features_train, features_test, labels_train)

# how many are predicted to be in the Chris -> value: 1

# I have also considered this version of the lambda -> value, item:(value, value + 1)[item == 1]
# but afer consideration and some reading on stack's.. conlcuded that using the form
# below is more future proof.
#[on_true] if [cond] else [on_false]

total_chris = reduce(lambda value, item: value + 1 if item == 1 else value, labels_predicted)

print "total_chris: ", total_chris

print "no. of Chris predicted:", sum(labels_predicted)
print "no. of Sara predicted:", len(labels_predicted)-sum(labels_predicted)

#print "10th : ", labels_predicted[10]

"""
# C range from 10 (10^1) to 10000 (10^4)
C_range = [10**1, 10**2, 10**3, 10**4]
for C in C_range:
    calc_accuracy(reducedTrainingSet, kernel, C, features_train, features_test, labels_train, labels_test)
"""

#########################################################
"""
- Results
1)Without Kernel parameters:
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time (fit): 1057.217 s
predict time: 114.834 s
accuracy_score time: 0.008 s
0.492036405006

2)With linear Kernel:
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time (fit): 168.396 s
predict time: 18.109 s
accuracy_score time: 0.001 s
0.984072810011

3) linear kernel + small/reduced trainig set
no. of Chris training emails: 7936
no. of Sara training emails: 7884
** Using reduced training set! **
training time (fit): 0.1 s
predict time: 1.014 s
accuracy_score time: 0.008 s
0.884527872582

4) rbf kernel + small/reduced trainig set
no. of Chris training emails: 7936
no. of Sara training emails: 7884
** Using reduced training set! **
training time (fit): 0.142 s
predict time: 1.256 s
accuracy_score time: 0.001 s
0.616040955631

5) rbf kernel + reduced trainig set + C RAnge of 10.0 to 10000
no. of Chris training emails: 7936
no. of Sara training emails: 7884

** Using reduced training set! **
C =  10  kernel =  rbf
training time (fit): 0.111 s
predict time: 1.11 s
accuracy_score time: 0.001 s
0.616040955631

** Using reduced training set! **
C =  100  kernel =  rbf
training time (fit): 0.108 s
predict time: 1.112 s
accuracy_score time: 0.001 s
0.616040955631

** Using reduced training set! **
C =  1000  kernel =  rbf
training time (fit): 0.104 s
predict time: 1.083 s
accuracy_score time: 0.001 s
0.821387940842

** Using reduced training set! **
C =  10000  kernel =  rbf
training time (fit): 0.1 s
predict time: 0.907 s
accuracy_score time: 0.001 s
0.892491467577

5) rbf kernel + reduced trainig set + C = 10000 (optimized)

no. of Chris training emails: 7936
no. of Sara training emails: 7884
C =  10000  kernel =  rbf
training time (fit): 115.415 s
predict time: 13.142 s
accuracy_score time: 0.001 s
0.990898748578

--------------------------------------------------------------------------------
What class does your SVM (0 or 1, corresponding to Sara and Chris respectively)
predict for element 10 of the test set? The 26th? The 50th?
Use the RBF kernel, C=10000, and 1% of the training set.

no. of Chris training emails: 7936
no. of Sara training emails: 7884
** Using reduced training set! **
C =  10000  kernel =  rbf
training time (fit): 0.098 s
predict time: 0.8 s
10th :  1 -> Chris
26th :  0 -> Sara
50th :  1 -> Chris

--------------------------------------------------------------------------------
There are over 1700 test events--how many are predicted to be in the
Chris (value:1) class?
Use the RBF kernel, C=10000., and the full training set.


1) with only 1% of the dataset:
no. of Chris training emails: 7936
no. of Sara training emails: 7884
** Using reduced training set! **
C =  10000  kernel =  rbf
training time (fit): 0.174 s
predict time: 1.323 s
total_chris:  1018

2) now with 100%

no. of Chris training emails: 7936
no. of Sara training emails: 7884
C =  10000  kernel =  rbf
training time (fit): 106.005 s
predict time: 10.463 s
total_chris:  877

3) iproved logging and some new learned tricks :)


"""
