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
sql_training_data_request = 'select IDENTIFIANT_PRODUIT, CATEGORIE_3, DESCRIPTION, LIBELLE from UNIFORMLY_RESTRAINED_TRAINING_DATA order by random() limit 100000'
cursor.execute(sql_training_data_request); 
 # retrieve the records from the database
fetched_training_data = cursor.fetchall()

training_identifiant_produit_list = [item[0] for item in fetched_training_data];
training_outputs  = [my_inverse_mapping_dictionnary[item[1]] for item in fetched_training_data];
training_documents  = [item[2] + ' '+item[3] for item in fetched_training_data];

print("%d documents" % len(training_documents))
print("%d categories" % len(training_outputs))

print("Loading 1000 samples randomly for testing... ")
sql_testing_data_request = 'select IDENTIFIANT_PRODUIT, CATEGORIE_3, DESCRIPTION, LIBELLE from UNIFORMLY_RESTRAINED_TRAINING_DATA order by random() limit 1000'
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

# here comes the deep learning modelisation
# getting the training and testing set from the data

    # Load the punkt tokenizer
#    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')

# ****** Split the labeled and unlabeled training sets into clean sentences
#
sentences = []  # Initialize an empty list of sentences
counter=0
for review in training_documents:
    sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
    counter=counter+1
    print('Adding sentences : '+str(counter))

for test_review in testing_documents:
    sentences += KaggleWord2VecUtility.review_to_sentences(test_review, tokenizer)
    counter=counter+1
    print('Adding sentences : '+str(counter))
#    print "Parsing sentences from unlabeled set"
#    for review in unlabeled_train["review"]:
#        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
#        counter=counter+1
#        print('Adding sentences : '+str(counter))

    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

    # Low performance : Set values for various parameters
#    num_features = 300    # Word vector dimensionality
#    min_word_count = 40   # Minimum word count
#    num_workers = 4       # Number of threads to run in parallel
#    context = 10          # Context window size
#    downsampling = 1e-3   # Downsample setting for frequent words

    # Low performance : Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 16      # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-2   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
model = Word2Vec(sentences, workers=num_workers,size=num_features, min_count = min_word_count, window = context, sample = downsampling, seed=1)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "1000features_100minwords_10context"
model.save(model_name)


trainDataVecs = getAvgFeatureVecs(getCleanDescriptions(training_documents), model, num_features )
trainDataOutput = training_outputs
print("Creating average feature vecs for test reviews")

testDataVecs = getAvgFeatureVecs(getCleanDescriptions(testing_documents), model, num_features )
testDataOutput= testing_outputs

###############################################################################
# Benchmark classifiers
def benchmark(cursor, clf_class, name):

    t0 = time()
    clf = clf_class.fit(trainDataVecs, trainDataOutput)
    print("done in %fs" % (time() - t0))

    if hasattr(clf, 'coef_'):
        print("Percentage of non zeros coef: %f"
              % (np.mean(clf.coef_ != 0) * 100))
    print("Predicting the outcomes of the testing set")
    t0 = time()
    pred = clf.predict(testDataVecs)
    print("done in %fs" % (time() - t0))

    print("Classification report on test set for classifier:")
    print(clf)
    print()
    #print(classification_report(testDataOutput, pred,
    #                            target_names=testing_names))
    for test_to_find, prediction in zip(testDataOutput, pred):
        counter = 0 
        if test_to_find == prediction :
            counter = counter+1
    print('Classification rate for: '+name+' '+str(counter) + '/'+str(len(testDataOutput)))
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
