'''
Created on 26 May 2015

@author: sduprey
'''

from __future__ import print_function

import psycopg2

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

print("Loading 1000 samples randomly for training... ")
sql_training_data_request = 'select IDENTIFIANT_PRODUIT, CATEGORIE_3, DESCRIPTION, LIBELLE from TRAINING_DATA order by random() limit 100000'
conn_string = "host='localhost' dbname='CATEGORIZERDB' user='postgres' password='mogette'"
# print the connection string we will use to connect
 
# get a connection, if a connect cannot be made an exception will be raised here
conn = psycopg2.connect(conn_string)
conn.autocommit = True
# conn.cursor will return a cursor object, you can use this cursor to perform queries
cursor = conn.cursor()
cursor.execute(sql_training_data_request); 
 # retrieve the records from the database
fetched_training_data = cursor.fetchall()

training_identifiant_produit_list = [item[0] for item in fetched_training_data];
training_outputs  = [item[1] for item in fetched_training_data];
training_documents  = [item[2] + ' '+item[3] for item in fetched_training_data];

data_train_target=training_outputs
# parsing all the stop words
stopwordsSet = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', file('/home/sduprey/My_Data/My_Semantics_Data/stopwords_fr.txt').read().lower()) 
stopwordsSet = [ unicode(word, "utf-8") for word in stopwordsSet]

data_train_data=[]
print("Cleaning and parsing the training set...\n")
#for i in xrange( 0, len(training_documents)):
counter=1
for training_doc in training_documents:
    print(' Cleaning Training Row '+str(counter))
    counter=counter+1
    wordList = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', unicode(training_doc.lower(), "utf-8")) 
    filtered_words = [w for w in wordList if not w in stopwordsSet]
    data_train_data.append(filtered_words)

print("Loading 100 samples randomly for testing... ")

sql_testing_data_request = 'select IDENTIFIANT_PRODUIT, CATEGORIE_3, DESCRIPTION, LIBELLE from TRAINING_DATA order by random() limit 1000'
cursor.execute(sql_testing_data_request); 
 # retrieve the records from the database
fetched_testing_data = cursor.fetchall()
testing_identifiant_produit_list = [item[0] for item in fetched_testing_data];
testing_outputs  = [item[1] for item in fetched_testing_data];
testing_documents  = [item[2] + ' '+item[3] for item in fetched_testing_data];

print("Predicting the labels of the test set...")
print("%d documents" % len(testing_documents))
print("%d categories" % len(testing_outputs))
data_test_target=testing_outputs
data_test_data=[]
print("Cleaning and parsing the training set...\n")
#for i in xrange( 0, len(training_documents)):
counter=1
for testing_doc in testing_documents:
    print(' Cleaning Testing Row '+str(counter))
    counter=counter+1
    wordList = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', unicode(testing_doc.lower(), "utf-8")) 
    filtered_words = [w for w in wordList if not w in stopwordsSet]
    data_test_data.append(filtered_words)


def size_mb(docs):
    return sum(len(s) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train_data)
data_test_size_mb = size_mb(data_test_data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train_data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test_data), data_test_size_mb))

# split a training set and a test set
X_train, X_test = data_train_data, data_test_data
y_train, y_test = data_train_target, data_test_target

def benchmark(clf,name):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    print("Saving into database")
    save_my_data(cursor, name, testing_identifiant_produit_list, y_test, pred)
    clf_descr = str(clf).split('(')[0]
    return clf_descr, train_time, test_time

def save_my_data(cursor, name, testing_identifiant_produit_list, y_test, pred): 
    my_lists = zip(testing_identifiant_produit_list, y_test, pred.tolist())
    print('saving into '+name)
    [insert_into_database(name, cursor,my_list_item[0],my_list_item[1],my_list_item[2]) for my_list_item in my_lists];

def insert_into_database(name, cursor, identifiant_produit, real_category, prediction) : 
    sql_string = 'INSERT INTO '+name+' (IDENTIFIANT_PRODUIT,REAL_CATEGORY,PREDICTION) VALUES (%s,%s,%s)'
    cursor.execute(sql_string, (identifiant_produit, real_category, prediction))


names = ["DL_KNN", "DL_NB", "DL_DT", "DL_AB", "DL_RF", "DL_RBF_SVM", "DL_LDA","DL_L_SVM", "DL_QDA"]
classifiers = [KNeighborsClassifier(2), GaussianNB(), DecisionTreeClassifier(max_depth=5), AdaBoostClassifier(),  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), SVC(gamma=2, C=1), LDA(), SVC(kernel="linear", C=0.025),QDA()]
    # iterate over classifiers
for name, clf in zip(names, classifiers):
    benchmark(clf,name)