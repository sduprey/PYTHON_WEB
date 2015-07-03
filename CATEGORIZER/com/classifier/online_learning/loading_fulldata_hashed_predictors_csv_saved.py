'''
Created on 25 Jun 2015

@author: sduprey
'''

from __future__ import print_function

import numpy as np

from scipy import sparse


import csv

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



my_full_hashed_training_matrix_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_training_matrix.csv'
indices = []
rowDatas = []
rowNumber = 0
for t, line in enumerate(open(my_full_hashed_training_matrix_file)):
    if t%3 == 0 :
        print('Row number '+line)
        rowNumber = int(line.rstrip()) 
#        for m, feat in enumerate(line.rstrip().split(',')):
#            print(feat)   
    if t%3 == 1 :
        print("My indices")
        indices =  [int(l) for l in line.rstrip().split(',')] 
#        for m, feat in enumerate(line.rstrip().split(',')):
#            print(feat)
    if t%3 == 2 :
        print('My row data '+line)
        rowDatas =  [float(l) for l in line.rstrip().split(',')] 
        print(rowDatas)
#        for m, feat in enumerate(line.rstrip().split(',')):
#            print(feat)
        
    
    
###### 

######output_file = open(my_full_hashed_training_matrix, 'w')
######my_writer = csv.writer(output_file)

######Xtrain = load_sparse_csr(my_saving_training_matrix_file)
####### number of training rows
######nrRows=Xtrain.shape[0]
######nrCols=Xtrain.shape[1]
######maxrowind=[]
######for i in range(nrRows):
######    r = Xtrain.getrow(i)# r is 1xA.shape[1] matrix
######    for j in range(nrCols):
        
######       data = [['Me', 'You'],\
######                ['293', '219'],\
######                ['54', '13']]
    

######    my_writer.writerows(data)
######    print(r)
    
    
######    maxrowind.append( r.indices[r.data.argmax()] if r.nnz else 0)

######  output_file.close()



######my_saving_testing_matrix_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_testing_matrix.bin.npz'
######Xtest = load_sparse_csr(my_saving_testing_matrix_file)
######print('test')

######my_saving_training_outputvector_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_data_training_output_vector.bin.npz'
######ytrain = load_output_vector(my_saving_training_outputvector_file)

