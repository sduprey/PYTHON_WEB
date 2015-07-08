'''
Created on 3 Jul 2015

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

import csv

from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


def save_output(filename,vector):
    np.savez(filename,data =vector)
    
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



print("Loading category mapping data")
sql_category_mapping_request = "select * from category_mapping"
cursor.execute(sql_category_mapping_request); 
category_mapping_data = cursor.fetchall()


my_mapping_dictionnary = {}

for mapping_item in category_mapping_data :
    my_mapping_dictionnary[mapping_item[0]]=mapping_item[1]

print("Loading cleaned samples for training... ")
sql_training_data_request = 'select IDENTIFIANT_PRODUIT, CATEGORIE_3, DESCRIPTION, LIBELLE, MARQUE from TRAINING_DATA'

cursor.execute(sql_training_data_request); 
fetched_training_data = cursor.fetchall()

training_identifiant_produit_list = [item[0] for item in fetched_training_data];
training_outputs  = [item[1] for item in fetched_training_data];

data_train_target=training_outputs
# split a training set and a test set
y_train = [ my_mapping_dictionnary[k] for k in data_train_target]

my_saving_training_outputvector_file_path = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_data_training_output_vector.csv'

my_saving_training_outputvector_file = open(my_saving_training_outputvector_file_path, 'w')
my__saving_training_outputvector_writer = csv.writer(my_saving_training_outputvector_file)

# number of training rows
nrRows=len(y_train)
for c, i in enumerate(range(nrRows)):
    print("Writing rows number "+ str(c)+" over "+ str(nrRows))
    value_row_i = y_train[i]
    my__saving_training_outputvector_writer.writerow([value_row_i])

my_saving_training_outputvector_file.close()
