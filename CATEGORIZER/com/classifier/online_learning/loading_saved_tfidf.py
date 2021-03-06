'''
Created on 25 Jun 2015

@author: sduprey
'''

from __future__ import print_function

import numpy as np

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


my_saving_training_outputvector_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/whole_tfidf_restrained_data_training_output_vector.bin.npz'
ytrain = load_output_vector(my_saving_training_outputvector_file)

    
my_saving_training_matrix_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/whole_tfidf_training_matrix.bin.npz'
Xtrain = load_sparse_csr(my_saving_training_matrix_file)

my_saving_testing_matrix_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/whole_tfidf_testing_matrix.bin.npz'
Xtest = load_sparse_csr(my_saving_testing_matrix_file)
print('test')


