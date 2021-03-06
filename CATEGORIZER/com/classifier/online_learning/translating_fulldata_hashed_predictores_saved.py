'''
Created on 25 Jun 2015

@author: sduprey
'''

from __future__ import print_function

import numpy as np
import csv
from scipy import sparse

def size_mb(docs):
    return sum(len(s) for s in docs) / 1e6

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def load_output_vector(filename):
    loader = np.load(filename)
    return loader['data']


my_saving_training_matrix_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_training_matrix.bin.npz'
my_full_hashed_training_matrix = '/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_training_matrix.csv'
my_full_hashed_training_output_file = open(my_full_hashed_training_matrix, 'w')
my_full_hashed_training_writer = csv.writer(my_full_hashed_training_output_file)
Xtrain = load_sparse_csr(my_saving_training_matrix_file)
# number of training rows
nrRows=Xtrain.shape[0]
nrCols=Xtrain.shape[1]
for c, i in enumerate(range(nrRows)):
    print("Writing rows number "+ str(c)+" over "+ str(nrRows))
    r = Xtrain.getrow(i)# r is 1xA.shape[1] matrix
    my_full_hashed_training_writer.writerow([i])
    my_full_hashed_training_writer.writerow(r.indices)
    my_full_hashed_training_writer.writerow(r.data)
#    counter=0
#    data=[0.]*nrRows
#    for row_indice in r.indices:
#        data[row_indice] = r.data[counter]
#        counter = counter + 1
#    my_writer.writerow(data)
my_full_hashed_training_output_file.close()


my_saving_testing_matrix_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_testing_matrix.bin.npz'
my_full_hashed_testing_matrix = '/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_testing_matrix.csv'
my_full_hashed_testing_output_file = open(my_full_hashed_testing_matrix, 'w')
my_full_hashed_testing_writer = csv.writer(my_full_hashed_testing_output_file)

Xtest = load_sparse_csr(my_saving_testing_matrix_file)
# number of training rows
nrtestRows=Xtest.shape[0]
nrtestCols=Xtest.shape[1]
for c, i in enumerate(range(nrtestRows)):
    print("Writing rows number "+ str(c)+" over "+ str(nrtestRows))
    r = Xtest.getrow(i)# r is 1xA.shape[1] matrix
    my_full_hashed_testing_writer.writerow([i])
    my_full_hashed_testing_writer.writerow(r.indices)
    my_full_hashed_testing_writer.writerow(r.data)   
my_full_hashed_testing_output_file.close()

#my_saving_training_outputvector_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_data_training_output_vector.bin.npz'
#ytrain = load_output_vector(my_saving_training_outputvector_file)

