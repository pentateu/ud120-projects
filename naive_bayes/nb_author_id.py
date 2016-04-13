#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
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

#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

# flag to reduce the training set to 1%
reducedTrainingSet = 1

if reducedTrainingSet:
    features_train = features_train[:len(features_train)/100]
    labels_train = labels_train[:len(labels_train)/100]
    print "** Using reduced training set! **"

t0 = time()
classifier.fit(features_train, labels_train)
print "training time (fit):", round(time()-t0, 3), "s"

t0 = time()
labels_predicted = classifier.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score

t0 = time()
accuracy = accuracy_score(labels_test, labels_predicted)
print "accuracy_score time:", round(time()-t0, 3), "s"

print accuracy
#########################################################
"""
- Results
1) Full training set
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time (fit): 1.529 s
predict time: 0.198 s
accuracy_score time: 0.001 s
0.973265073948

2) Reduced training set (1%)
no. of Chris training emails: 7936
no. of Sara training emails: 7884
** Using reduced training set! **
training time (fit): 0.011 s
predict time: 0.192 s
accuracy_score time: 0.001 s
0.919226393629

"""
