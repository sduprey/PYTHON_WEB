'''
Created on 22 May 2015

@author: sduprey
'''

from __future__ import print_function

from time import time
import psycopg2
import numpy as np
import scipy.sparse as sp
import pylab as pl

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

print(__doc__)
# we train our data over 100 000  
print("Loading 100000 samples randomly for training... ")
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

print("%d documents" % len(training_documents))
print("%d categories" % len(training_outputs))

stopwordsSet = re.findall('[a-z]+', file('/home/sduprey/My_Data/My_Semantics_Data/stopwords_fr.txt').read().lower()) 

print("Extracting features from the dataset using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(encoding='utf-8',stop_words=stopwordsSet)

X_train = vectorizer.fit_transform(training_documents)
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X_train.shape)
assert sp.issparse(X_train)
y_train = training_outputs

print("Loading 1000 samples randomly for testing... ")

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

print("Extracting features from the dataset using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(testing_documents)
y_test = testing_outputs
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X_test.shape)


###############################################################################
# Benchmark classifiers
def benchmark(cursor, clf_class, params, name):
    print("parameters:", params)
    t0 = time()
    clf = clf_class(**params).fit(X_train, y_train)
    print("done in %fs" % (time() - t0))

    if hasattr(clf, 'coef_'):
        print("Percentage of non zeros coef: %f"
              % (np.mean(clf.coef_ != 0) * 100))
    print("Predicting the outcomes of the testing set")
    t0 = time()
    pred = clf.predict(X_test)
    print("done in %fs" % (time() - t0))

    print("Classification report on test set for classifier:")
    print(clf)
    print()
    print(classification_report(y_test, pred,
                                target_names=testing_outputs))

    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:")
    print(cm)
    print("Saving data to database:")
    save_my_data(cursor, name, testing_identifiant_produit_list, y_test, pred)


    # Show confusion matrix
    pl.matshow(cm)
    pl.title('Confusion matrix of the %s classifier' % name)
    pl.colorbar()

def save_my_data(cursor, name, testing_identifiant_produit_list, y_test, pred): 
    my_lists = zip(testing_identifiant_produit_list, y_test, pred.tolist())
    print('saving into '+name) 
    [insert_into_database(name, cursor,my_list_item[0],my_list_item[1],my_list_item[2]) for my_list_item in my_lists];
      
                

def insert_into_database(name, cursor, identifiant_produit, real_category, prediction) : 
    sql_string = 'INSERT INTO '+name+' (IDENTIFIANT_PRODUIT,REAL_CATEGORY,PREDICTION) VALUES (%s,%s,%s)'
    cursor.execute(sql_string, (identifiant_produit, real_category, prediction))
    
      
from itertools import chain
def fetch_data(*args):
    return list(chain.from_iterable(zip(*args)))

print("Testbenching a linear classifier...")
parameters = {
    'loss': 'hinge',
    'penalty': 'l2',
    'n_iter': 50,
    'alpha': 0.00001,
    'fit_intercept': True,
}

benchmark(cursor, SGDClassifier, parameters, 'NOT_HASHED_SGD')

print("Testbenching a MultinomialNB classifier...")
parameters = {'alpha': 0.01}

benchmark(cursor, MultinomialNB, parameters, 'NOT_HASHED_MULTINOMIALNB')

pl.show()