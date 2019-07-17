#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:51:42 2019

@author: aman
"""
#importing the package 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the iris dataset from subpackage name as dataset from sklearn package
from sklearn.datasets import load_iris
dataset=load_iris()

#creating the features as X
X=dataset.data
#Spliting the lable y 
y = dataset.target

#spliting the dataset into traning and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

#importing Desion tree Classifier
from sklearn.tree import DecisionTreeClassifier
#creating the object of Decision Tree and max depth is that how much node of a tree you want
dtf=DecisionTreeClassifier(max_depth=2)
#fitting the model into training dataset
dtf.fit(X_train,y_train)

#preict the value on testing
y_pred=dtf.predict(X_test)

dtf.score(X_test,y_test)

#creating the dot file of a tree so that we can see how the tree of this dataset look like
#by using graphviz package
from sklearn.tree import export_graphviz
export_graphviz(dtf,out_file="tree.dot")



#opening the tree.dotfie
import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)



