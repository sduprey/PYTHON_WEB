'''
Created on 2 Jul 2015

@author: sduprey
'''
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import numpy as np

row  = np.array([0, 3, 1, 0])
col  = np.array([0, 2, 3, 2])
data = np.array([-3, 4, 11, -7])
A= csr_matrix((data, (row, col)), shape=(4, 4))
print A.toarray()

nrRows=A.shape[0]
maxrowind=[]
for i in range(nrRows):
    r = A.getrow(i)# r is 1xA.shape[1] matrix
    maxrowind.append( r.indices[r.data.argmax()] if r.nnz else 0)
print maxrowind 




row  = np.array([0, 3, 1, 0])
col  = np.array([0, 2, 3, 2])
data = np.array([-3, 4, 11, -7])
A= coo_matrix((data, (row, col)), shape=(4, 4))
print A.toarray()

nrRows=A.shape[0]
maxrowind=[]
for i in range(nrRows):
    r = A.getrow(i)# r is 1xA.shape[1] matrix
    maxrowind.append( r.indices[r.data.argmax()] if r.nnz else 0)
print maxrowind 