#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 23:55:30 2018

@author: animesh
"""

from sklearn import tree
from sklearn.metrics import accuracy_score

import numpy as np
#from scipy.io import arff
def load_data():
    #dataset = arff.load(open('dataset.arff', 'rb'))
    #train = np.array(dataset['data'])
    #train = np.load('dataset.arff', delimiter=',', dtype=np.int32)
    #train = pd.read_csv('dataset.csv')
    
    train= np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)
    X = train[:, :-1]
    Y = train[:, -1]
    
    x_train = X[:2000]
    y_train = Y[:2000]
    
    x_test = X[2000:]
    y_test = Y[2000:]
    
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    print("Decision Tree Classifier")
    
    #load traiing data
    x_train, y_train, x_test, y_test = load_data()
    
    #create classifier
    classifier = tree.DecisionTreeClassifier()
    print("decision tree created")
    
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