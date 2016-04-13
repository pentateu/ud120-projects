#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
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
def predict(one_percent, min_samples_split, features_train, features_test, labels_train):
    from sklearn import tree

    if one_percent:
        features_train = features_train[:len(features_train)/100]
        labels_train = labels_train[:len(labels_train)/100]
        print "** Using reduced training set! **"

    print "min_samples_split = ", min_samples_split

    classifier = tree.DecisionTreeClassifier(min_samples_split=min_samples_split)

    t0 = time()
    classifier.fit(features_train, labels_train)
    print "training time (fit):", round(time()-t0, 3), "s"

    t0 = time()
    labels_predicted = classifier.predict(features_test)
    print "predict time:", round(time()-t0, 3), "s"

    return labels_predicted

def calc_accuracy(one_percent, min_samples_split, features_train, features_test, labels_train, labels_test):
    from sklearn.metrics import accuracy_score

    labels_predicted = predict(one_percent, min_samples_split, features_train, features_test, labels_train)

    t0 = time()
    accuracy = accuracy_score(labels_test, labels_predicted)
    print "accuracy_score time:", round(time()-t0, 3), "s"

    print "accuracy: ", accuracy

    return accuracy

### 1) - What is the accuracy?
### min_samples_split=40
calc_accuracy(0, 40, features_train, features_test, labels_train, labels_test)


#########################################################
#########################################################
"""
- Results
### 1) - What is the accuracy?
### min_samples_split=40



"""
