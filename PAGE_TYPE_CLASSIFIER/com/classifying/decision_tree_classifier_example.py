'''
Created on 26 Jan 2015

@author: sduprey
'''
from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
print cross_val_score(clf, iris.data, iris.target, cv=10)

#array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
#        0.93...,  0.93...,  1.     ,  0.93...,  1.      ])