# -*- coding: utf-8 -*-
'''
Created on 16 Jun 2015

@author: sduprey
'''

import psycopg2
import pandas as pd
from nltk.corpus import stopwords
import nltk.data
from KaggleWord2VecUtility import KaggleWord2VecUtility
import re

if __name__ == '__main__':

#    train = pd.read_csv( os.path.join(os.path.dircdiscount_uniformly_restrained_training.csvname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
#    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
#    unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
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

    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
     " end of file \n" % (train["description"].size,
     test["description"].size )



 

    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences
    counter=0
    print "Parsing sentences from training set"
    for my_description in train["description"]:
        wordList = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', my_description.lower()) 
        sentences += wordList
        counter=counter+1
        print('Adding sentences : '+str(counter))

    print "Parsing sentences from testing set"
    for test_review in test["description"]:
        wordList = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', my_description.lower()) 
        sentences += wordList
        counter=counter+1
        print('Adding sentences : '+str(counter))