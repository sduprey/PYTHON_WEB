# -*- coding: utf-8 -*-
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
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
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

conn_string = "host='localhost' dbname='CATEGORIZERDB' user='postgres' password='mogette'"
# print the connection string we will use to connect
 
# get a connection, if a connect cannot be made an exception will be raised here
conn = psycopg2.connect(conn_string)
conn.autocommit = True
# conn.cursor will return a cursor object, you can use this cursor to perform queries
cursor = conn.cursor()

print("Loading cleaned samples for training... ")
sql_training_data_request = 'select IDENTIFIANT_PRODUIT, CATEGORIE_3, DESCRIPTION, LIBELLE, MARQUE from TRAINING_DATA'

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
    sentence_to_add = " ".join(wordList)
    data_train_data.append(sentence_to_add)


print("Loading all testing... ")

sql_testing_data_request = 'select identifiant_produit, description, libelle, marque, prix from TESTING_DATA'
cursor.execute(sql_testing_data_request); 
 # retrieve the records from the database
fetched_testing_data = cursor.fetchall()
testing_identifiant_produit_list = [item[0] for item in fetched_testing_data];
testing_documents  = [item[1] + ' '+item[2] for item in fetched_testing_data];

print("Predicting the labels of the test set...")
print("%d documents" % len(testing_documents))
data_test_data=[]
print("Cleaning and parsing the training set...\n")
#for i in xrange( 0, len(training_documents)):
counter=1
for testing_doc in testing_documents:
    print(' Cleaning Testing Row '+str(counter))
    counter=counter+1
    wordList = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', unicode(testing_doc.lower(), "utf-8")) 
    filtered_words = [w for w in wordList if not w in stopwordsSet]
    sentence_to_add = " ".join(wordList)
    data_test_data.append(sentence_to_add)

def size_mb(docs):
    return sum(len(s) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train_data)
data_test_size_mb = size_mb(data_test_data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train_data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test_data), data_test_size_mb))

# split a training set and a test set
y_train = data_train_target



print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
opts.use_hashing=True
if opts.use_hashing:
    # old version :vectorizer = HashingVectorizer(stop_words=stopwordsSet, non_negative=True, n_features=opts.n_features)   
    vectorizer = HashingVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None)
    X_train = vectorizer.transform(data_train_data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words=stopwordsSet)
    X_train = vectorizer.fit_transform(data_train_data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test_data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)



###############################################################################
# Benchmark classifiers
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

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("Saving data to database:")
    save_results_data(cursor, name, testing_identifiant_produit_list, pred)
    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr,train_time,test_time

def save_results_data(cursor, name, testing_identifiant_produit_list, pred): 
    my_lists = zip(testing_identifiant_produit_list, pred.tolist())
    print('saving into '+name)
    [insert_into_database(name, cursor,my_list_item[0],my_list_item[1]) for my_list_item in my_lists];

def insert_into_database(name, cursor, identifiant_produit,prediction) : 
    sql_string = 'INSERT INTO '+name+' (IDENTIFIANT_PRODUIT,PREDICTION) VALUES (%s,%s)'
    cursor.execute(sql_string, (identifiant_produit,prediction))
    
benchmark(LinearSVC(), "PRODUCTION_CLEANED_LINEARSVCCLASS")
benchmark(SVC(gamma=2, C=1), "PRODUCTION_CLEANED_SVCRBF")
benchmark(PassiveAggressiveClassifier(n_iter=100), "PRODUCTION_CLEANED_PASSIVEAGGRESSIVE")
benchmark(SGDClassifier(alpha=.0001, n_iter=100, penalty="l1"), "PRODUCTION_CLEANED_SGDL1")

benchmark(SGDClassifier(alpha=.0001, n_iter=100, penalty="l2"), "PRODUCTION_CLEANED_SGDL2")
benchmark(SGDClassifier(alpha=.0001, n_iter=100,  penalty="elasticnet"),"PRODUCTION_CLEANED_ELASTICNET")
benchmark(BernoulliNB(alpha=.01),'PRODUCTION_CLEANED_BERNOULLINB')
benchmark(NearestCentroid(),'PRODUCTION_CLEANED_NEARESTCENTROID')
benchmark(KNeighborsClassifier(n_neighbors=10), "PRODUCTION_CLEANED_KNN")
benchmark(MultinomialNB(alpha=.01),'PRODUCTION_CLEANED_MULTINOMIALNB')

benchmark(DecisionTreeClassifier(), "PRODUCTION_CLEANED_DECISIONTREE") 
benchmark(RandomForestClassifier(), "PRODUCTION_CLEANED_RANDOMFOREST")  
benchmark(AdaBoostClassifier(), "PRODUCTION_CLEANED_ADABOOSTCLASSIFIER")
