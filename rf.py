#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:21:05 2018

@author: animesh
"""


from sklearn.metrics import accuracy_score

import numpy as np
#from scipy.io import arff
def load_data():
   
    
    train= np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)
    X = train[:, :-1]
    Y = train[:, -1]
    
    x_train = X[:2000]
    y_train = Y[:2000]
    
    x_test = X[2000:]
    y_test = Y[2000:]
    
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    print("Random Forest Classifier")
    
    #load traiing data
    x_train, y_train, x_test, y_test = load_data()
    
    #create classifier
    from sklearn.ensemble import RandomForestClassifier
    #classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier = RandomForestClassifier()
    #classifier = tree.DecisionTreeClassifier()
    #rint("decision tree created")
    
    #train model
    print("model training")
    classifier.fit(x_train, y_train)
    
    #test model
    y_pred = classifier.predict(x_test)
    print("prediction done")
    
    accuracy = 100 * accuracy_score(y_test,y_pred)
    print("accuracy = " + str(accuracy))
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)