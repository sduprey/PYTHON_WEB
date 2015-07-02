#!/usr/bin/env python

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# *************************************** #

import os
import psycopg2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

if __name__ == '__main__':
    conn_string = "host='localhost' dbname='CATEGORIZERDB' user='postgres' password='mogette'"
    # print the connection string we will use to connect
    print "Connecting to database\n    ->%s" % (conn_string)
 
    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)
    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor()
    # execute our Query
    # X = np.asarray(predictors_list);
    my_idmapping_request = "select * from CATEGORY_MAPPING"

   # fetching data to display for magasin Musique
    cursor.execute(my_idmapping_request); 
    mapping_data = cursor.fetchall()
    my_mapping_dictionnary = {}
    for element in mapping_data:
        my_mapping_dictionnary[element[1]]=element[0]
    training_data_path = '/home/sduprey/My_Data/My_Cdiscount_Challenge/heavy_cdiscount_uniformly_restrained_training.csv'
    testing_data_path = '/home/sduprey/My_Data/My_Cdiscount_Challenge/cdiscount_uniformly_restrained_testing.csv'

    # Read data from files
    train = pd.read_csv(training_data_path, header=0, delimiter=";", quoting=3 )
    test = pd.read_csv(testing_data_path, header=0, delimiter=";", quoting=3 )


    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print "Cleaning and parsing the training set...\n"
    for i in xrange( 0, len(train["description"])):
        print 'Row '+str(i)
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["description"][i], True)))


    # ****** Create a bag of words from the training set
    #
    print "Creating the bag of words...\n"


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
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

    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0,len(test["review"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()


     # we here list the different classifiers we want to test
    names = ["Nearest Neighbors", "Naive Bayes", "Decision Tree", "AdaBoost", "Random Forest", "RBF SVM", "LDA","Linear SVM", "QDA"]
    classifiers = [
    KNeighborsClassifier(2),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    AdaBoostClassifier(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    SVC(gamma=2, C=1),
    LDA(),
    SVC(kernel="linear", C=0.025),
    QDA()]
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print "Training " + name +" to calibrate our classifier"

        clf.fit(train_data_features, train["output"])
        #score = clf.score(X_test, y_test)
        print name+" has been trained, we now make the prediction"

        Id_Categorie=[]
        # Test & extract results
        result = clf.predict(test_data_features)
        for element in result.flat:
            Id_Categorie.append(my_mapping_dictionnary[element])
#        print element
        # Write the test results
        output = pd.DataFrame( data={"Id_Produit":test["id"], "Id_Categorie":Id_Categorie} )
        output.to_csv('/home/sduprey/My_Data/My_Outgoing_Data/My_Cdiscount_Challenge/heavy_'+ name +'bagofword_AverageVectors.csv', index=False, quoting=3)
        print 'Wrote /home/sduprey/My_Data/My_Outgoing_Data/My_Cdiscount_Challenge/heavy_'+ name +'bagofword_AverageVectors.csv'


