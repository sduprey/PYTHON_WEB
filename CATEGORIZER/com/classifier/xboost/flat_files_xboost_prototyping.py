# -*- coding: utf-8 -*-
'''
Created on 11 Jun 2015

@author: sduprey
'''
#!/usr/bin/env python

#
# *************************************** #


# ****** Read the two training sets and the test set
#

import pandas as pd
import numpy as np  # Make sure that numpy is imported


from time import time
import re
import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.cross_validation import train_test_split

from sklearn import metrics

def size_mb(docs):
    return sum(len(s) for s in docs) / 1e6

###############################################################################
# Benchmark classifiers
def benchmark_gradient_boosting(name):
    print('_' * 80)
    print("Training XGBOOST classifier: ")

    t0 = time()
    xgb_model = xgb.XGBClassifier().fit(X_train,y_train)
    pred = xgb_model.predict(X_train)

    training = time() - t0
    print("training time:  %0.3fs" % training)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print_report = False
    if print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred))
#        print(metrics.classification_report(y_test, pred,
#                                            target_names=categories))
    print_cm = False
    if print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
    
    performance_assessment( name, y_test, pred)
    write_file( name, y_test, pred)

def performance_assessment(name,y_test, pred):
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
#    print("classification report:")
#    print(metrics.classification_report(y_test, pred))
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))


def write_file( name, testing_identifiant_produit_list, y_test, pred): 
    my_lists = zip(testing_identifiant_produit_list, y_test, pred.tolist())
    print('saving into '+name)
    [insert_into_file(name,my_list_item[0],my_list_item[1],my_list_item[2]) for my_list_item in my_lists];

def insert_into_file(name, identifiant_produit, real_category, prediction) : 
    sql_string = 'INSERT INTO '+name+' (IDENTIFIANT_PRODUIT,REAL_CATEGORY,PREDICTION) VALUES (%s,%s,%s)'
    #cursor.execute(sql_string, (identifiant_produit, real_category, prediction))
 

if __name__ == '__main__':
    
    category_mapping_path = '/home/sduprey/My_Data/My_Cdiscount_Challenge/category_mapping.csv'
    
    training_data_path = '/home/sduprey/My_Data/My_Cdiscount_Challenge/cdiscount_uniformly_restrained_training.csv'
    testing_data_path = '/home/sduprey/My_Data/My_Cdiscount_Challenge/cdiscount_uniformly_restrained_testing.csv'

    # Read data from files
    train = pd.read_csv(training_data_path, header=0, delimiter=";", quoting=3 )
    test = pd.read_csv(testing_data_path, header=0, delimiter=";", quoting=3 )

    #   train["description"] : libelle + long description
    #   train["output"] : id de la categorie de sortie
    #   train["output_string"] : string de la categorie de sortie
 
    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
     " end of file \n" % (train["description"].size,
     test["description"].size )

#     # mapping category strings and category ids
##    my_mapping_dictionnary_bis = {}
##    for j in range(train["description"].size):
##        my_mapping_dictionnary_bis[train["output"][j]] = train["output_string"][j] 
        
    # mapping category strings and category ids
    mapping_data_df =  pd.read_csv(category_mapping_path, header=0, delimiter=";", quoting=3 )
    my_dimensions = mapping_data_df.shape
    my_mapping_dictionnary = {}
    for i in range(my_dimensions[0]):
         my_mapping_dictionnary[mapping_data_df.iloc[i,0]]=mapping_data_df.iloc[i,1]

     # parsing all the stop words
    stopwordsSet = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', file('/home/sduprey/My_Data/My_Semantics_Data/stopwords_fr.txt').read().lower()) 
    stopwordsSet = [unicode(word, "utf-8") for word in stopwordsSet]

    data_train_data=[]
    print("Cleaning and parsing the training set...\n")
#for i in xrange( 0, len(training_documents)):
    counter=1
    for training_doc in train["description"]:
        print(' Cleaning Training Row '+str(counter))
        counter=counter+1
        wordList = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', unicode(training_doc.lower(), "utf-8")) 
        filtered_words = [w for w in wordList if not w in stopwordsSet]
        sentence_to_add = " ".join(wordList)
        data_train_data.append(sentence_to_add)

    data_test_data=[]
    print("Cleaning and parsing the training set...\n")
    counter=1
    for testing_doc in test["description"]:
        print(' Cleaning Testing Row '+str(counter))
        counter=counter+1
        wordList = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', unicode(testing_doc.lower(), "utf-8")) 
        filtered_words = [w for w in wordList if not w in stopwordsSet]
        sentence_to_add = " ".join(wordList)
        data_test_data.append(sentence_to_add)


    data_train_size_mb = size_mb(data_train_data)
    data_test_size_mb = size_mb(data_test_data)

    print("%d documents - %0.3fMB (training set)" % (len(data_train_data), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (len(data_test_data), data_test_size_mb))

    # split a training set and a test set
    y = train["output"]

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    use_hashing=False
    if use_hashing:
    # old version :vectorizer = HashingVectorizer(stop_words=stopwordsSet, non_negative=True, n_features=opts.n_features)   
        vectorizer = HashingVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None)
        X = vectorizer.transform(data_train_data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words=stopwordsSet)
        X = vectorizer.fit_transform(data_train_data)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# mapping from integer feature name to original token string    
    if use_hashing:
        feature_names = None
    else:
        feature_names = vectorizer.get_feature_names()

    feature_selection = False
    keep=10000
    if feature_selection:
        print("Extracting %d best features by a chi-squared test" % keep)
        t0 = time()
        shrinking_model = SelectKBest(f_classif, k=keep)
        X_train = shrinking_model.fit_transform(X_train, y_train)
        X_test = shrinking_model.transform(X_test)

    print("done in %fs" % (time() - t0))
    print()



    results = []
    results.append(benchmark_gradient_boosting('XGBOOST'))
