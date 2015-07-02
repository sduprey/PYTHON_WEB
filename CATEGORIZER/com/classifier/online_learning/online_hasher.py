# -*- coding: utf-8 -*-
'''
Created on 26 June 2015
@author: sduprey
'''

from __future__ import print_function
from collections import defaultdict
import re
import sys
from time import time

import numpy as np

import psycopg2


from sklearn.feature_extraction import DictVectorizer, FeatureHasher


def n_nonzero_columns(X):
    """Returns the number of non-zero columns in a CSR matrix X."""
    return len(np.unique(X.nonzero()[1]))


def tokens(doc):
    """Extract tokens from doc.

    This uses a simple regex to break strings into tokens. For a more
    principled approach, see CountVectorizer or TfidfVectorizer.
    """
    wordList = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', unicode(doc.lower(), "utf-8")) 
    filtered_words = [w for w in wordList if not w in stopwordsSet]
#    sentence_to_add = " ".join(wordList)
#    raw_data.append(sentence_to_add)
#    return (tok.lower() for tok in re.findall(r"\w+", doc))
    return filtered_words


def token_freqs(doc):
    """Extract a dict mapping tokens from doc to their frequencies."""
    freq = defaultdict(int)
    for tok in tokens(doc):
        freq[tok] += 1
    return freq

# Uncomment the following line to use a larger set (11k+ documents)
#categories = None

print(__doc__)
print("Usage: %s [n_features_for_hashing]" % sys.argv[0])
print("    The default number of features is 2**18.")
print()

try:
    n_features = int(sys.argv[1])
except IndexError:
    n_features = 2 ** 18
except ValueError:
    print("not a valid number of features: %r" % sys.argv[1])
    sys.exit(1)


print("Loading raw data from the training set")
conn_string = "host='localhost' dbname='CATEGORIZERDB' user='postgres' password='mogette'"
# print the connection string we will use to connect
 
# get a connection, if a connect cannot be made an exception will be raised here
conn = psycopg2.connect(conn_string)
conn.autocommit = True
# conn.cursor will return a cursor object, you can use this cursor to perform queries
cursor = conn.cursor()

print("Loading cleaned samples for training... ")
sql_training_data_request = 'select IDENTIFIANT_PRODUIT, CATEGORIE_3, DESCRIPTION, LIBELLE, MARQUE from UNIFORMLY_RESTRAINED_TRAINING_DATA'

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

#raw_data=[]#
#print("Cleaning and parsing the training set...\n")
##for i in xrange( 0, len(training_documents)):
#counter=1
#for training_doc in training_documents:
#    print(' Cleaning Training Row '+str(counter))
#    counter=counter+1
#    wordList = re.findall('[\\wàáâãäçèéêëœìíîïñòóôõöùúûüÿÁÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇŒ]+', unicode(training_doc.lower(), "utf-8")) 
#    filtered_words = [w for w in wordList if not w in stopwordsSet]
#    sentence_to_add = " ".join(wordList)
#    raw_data.append(sentence_to_add)


raw_data = training_documents
#data_size_mb = sum(len(s.encode('utf-8')) for s in raw_data) / 1e6
#print("%d documents - %0.3fMB" % (len(raw_data), data_size_mb))
#print()

print("DictVectorizer")
t0 = time()
vectorizer = DictVectorizer()
vectorizer.fit_transform(token_freqs(d) for d in raw_data)
duration = time() - t0
#print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))
print("Found %d unique terms" % len(vectorizer.get_feature_names()))
print()

print("FeatureHasher on frequency dicts")
t0 = time()
hasher = FeatureHasher(n_features=n_features)
X = hasher.transform(token_freqs(d) for d in raw_data)
duration = time() - t0
#print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))
print("Found %d unique terms" % n_nonzero_columns(X))
print()

print("FeatureHasher on raw tokens")
t0 = time()
hasher = FeatureHasher(n_features=n_features, input_type="string")
X = hasher.transform(tokens(d) for d in raw_data)
duration = time() - t0
#print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))
print("Found %d unique terms" % n_nonzero_columns(X))