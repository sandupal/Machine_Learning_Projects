#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:36:55 2019

@author: aman
"""
#importing the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

#loading the dataset
from sklearn.datasets import fetch_mldata
dataset=fetch_mldata("/home/aman/scikit_learn_data/mldata/mnist-original")

#spilting the features
X=dataset.data
#spliting the label
y = dataset.target

#lets have a image of a number written at this index of x
some_digit=X[51851]
#the image will be in 28*28 shape and but in dataset is in 784(28*28)
some_digit_image=some_digit.reshape(28,28)
#ploting the image
pt.imshow(some_digit_image)
pt.show()

#spliting the data in training and testing model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

#creating the decsion tree model
from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier(max_depth=10)
dtf.fit(X_train,y_train)

#checking the either prediction is correct or not
y_somedigit=dtf.predict(X[[6589,484,51851,8484,1451],:784])



#creating the dot file of a tree so that we can see how the tree of this dataset look like
#by using graphviz package
from sklearn.tree import export_graphviz
export_graphviz(dtf,out_file="tree1.dot")



#opening the tree.dotfie
import graphviz
with open("tree1.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)







