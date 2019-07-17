#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:22:29 2019

@author: aman
"""
#importing package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading the dataset from sklearn name as iris dataset
from sklearn.datasets import load_iris
dataset=load_iris()

#spliting the features
X = dataset.data
#spliting the label
y = dataset.target

#spliting the dataset to training model and testing model by 80 and 20 %
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


#importing the KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
#fitting the algorithm
knn.fit(X_train,y_train)

#takeprediction on test data and now we can compare the actual values and predicted value .

y_pred=knn.predict(X_test)

#chevking the score of the model
knn.score(X_test,y_test)