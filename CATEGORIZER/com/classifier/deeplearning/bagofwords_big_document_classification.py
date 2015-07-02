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

from sklearn.feature_extraction.text import CountVectorizer

import psycopg2
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from KaggleWord2VecUtility import KaggleWord2VecUtility

print(__doc__)
# here comes the data to train
# here comes the data to train
# here comes the data to train

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    if nwords == 0 :
        print('Trouble here where no words was found in the vocabulary')
    else :
        featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews


def getCleanDescriptions(descriptions):
    clean_descriptions = []
    local_counter=0
    for description in descriptions:
        clean_descriptions.append( KaggleWord2VecUtility.review_to_wordlist( description, remove_stopwords=True ))
        local_counter=local_counter+1
        print('Adding line : '+str(local_counter))
    return clean_descriptions


print("Connecting to the database ")
conn_string = "host='localhost' dbname='CATEGORIZERDB' user='postgres' password='mogette'"
# print the connection string we will use to connect
 
# get a connection, if a connect cannot be made an exception will be raised here
conn = psycopg2.connect(conn_string)
conn.autocommit = True
# conn.cursor will return a cursor object, you can use this cursor to perform queries
cursor = conn.cursor()

my_idmapping_request = "select * from CATEGORY_MAPPING"

print("Getting the mapping between category output and the id")
# fetching data to display for magasin Musique
cursor.execute(my_idmapping_request); 
mapping_data = cursor.fetchall()
my_mapping_dictionnary = {}
my_inverse_mapping_dictionnary = {}
for element in mapping_data:
    # my_mapping_dictionnary[id] ==> category name
    my_mapping_dictionnary[element[1]]=element[0]
    my_inverse_mapping_dictionnary[element[0]]=element[1]
        
print("Loading limited size data samples randomly for training... ")
sql_training_data_request = 'select IDENTIFIANT_PRODUIT, CATEGORIE_3, DESCRIPTION, LIBELLE from TRAINING_DATA order by random() limit 10000'
cursor.execute(sql_training_data_request); 
 # retrieve the records from the database
fetched_training_data = cursor.fetchall()

training_identifiant_produit_list = [item[0] for item in fetched_training_data];
training_outputs  = [my_inverse_mapping_dictionnary[item[1]] for item in fetched_training_data];
training_documents  = [item[2] + ' '+item[3] for item in fetched_training_data];

print("%d documents" % len(training_documents))
print("%d categories" % len(training_outputs))

print("Loading 1000 samples randomly for testing... ")
sql_testing_data_request = 'select IDENTIFIANT_PRODUIT, CATEGORIE_3, DESCRIPTION, LIBELLE from TRAINING_DATA order by random() limit 100'
cursor.execute(sql_testing_data_request); 
 # retrieve the records from the database
fetched_testing_data = cursor.fetchall()
testing_identifiant_produit_list = [item[0] for item in fetched_testing_data];
testing_outputs  = [my_inverse_mapping_dictionnary[item[1]] for item in fetched_testing_data];
testing_names  = [item[1] for item in fetched_testing_data];
testing_documents  = [item[2] + ' '+item[3] for item in fetched_testing_data];

print("Predicting the labels of the test set...")
print("%d documents" % len(testing_documents))
print("%d categories" % len(testing_outputs))

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list

print("Cleaning and parsing the training set...\n")
for i in xrange( 0, len(training_documents)):
     print('Row '+str(i))
     clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(training_documents[i], True)))
     vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
     train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
     train_data_features = train_data_features.toarray()

    # ******* Train a random forest using the bag of words
    #
 
    # Create an empty list and append the clean reviews one by one
     clean_test_reviews = []
     print("Cleaning and parsing the test set movie reviews...\n")
     for i in xrange(0,len(testing_documents)):
         clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(testing_documents[i], True)))

    # Get a bag of words for the test set, and convert to a numpy array
     test_data_features = vectorizer.transform(clean_test_reviews)
     test_data_features = test_data_features.toarray()

###############################################################################
# Benchmark classifiers
def benchmark(cursor, clf_class, name):

    t0 = time()
    clf = clf_class.fit(train_data_features, training_outputs)
    print("done in %fs" % (time() - t0))

    if hasattr(clf, 'coef_'):
        print("Percentage of non zeros coef: %f"
              % (np.mean(clf.coef_ != 0) * 100))
    print("Predicting the outcomes of the testing set")
    t0 = time()
    pred = clf.predict(test_data_features)
    print("done in %fs" % (time() - t0))

    print("Classification report on test set for classifier:")
    print(clf)
    print()
    #print(classification_report(testDataOutput, pred,
    #                            target_names=testing_names))
    for test_to_find, prediction in zip(testing_outputs, pred):
        counter = 0 
        if test_to_find == prediction :
            counter = counter+1
    print('Classification rate for: '+name+' '+str(counter) + '/'+str(len(testing_outputs)))
  #  print(classification_report(testDataOutput, pred))

  #  cm = confusion_matrix(testDataOutput, pred)
  #  print("Confusion matrix:")
  #  print(cm)
#  we are here just prototyping    
#    print("Saving data to database:")
#    save_my_data(cursor, name, testing_identifiant_produit_list, testDataOutput, pred)

    # Show confusion matrix
#    pl.matshow(cm)
  #  pl.title('Confusion matrix of the %s classifier' % name)
 #   pl.colorbar()

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

print("Test benching a few classifiers using word2vec training data")

names = ["Nearest Neighbors", "Naive Bayes", "Decision Tree", "AdaBoost", "Random Forest", "RBF SVM", "LDA","Linear SVM", "QDA"]
classifiers = [KNeighborsClassifier(2),GaussianNB(),DecisionTreeClassifier(max_depth=5),AdaBoostClassifier(),RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),SVC(gamma=2, C=1),LDA(),SVC(kernel="linear", C=0.025),QDA()]
# iterate over classifiers
for name, clf in zip(names, classifiers):
    benchmark(cursor, clf, name)
