'''
Created on 25 Jun 2015

@author: sduprey
'''
'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004
Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>
Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION
 0. You just DO WHAT THE FUCK YOU WANT TO.
'''


from datetime import datetime
from math import log, exp, sqrt,factorial
import pickle

# TL; DR
# the main learning process start at line 122


# parameters #################################################################

import sys
data_dir='/home/sduprey/My_Data/My_Cdiscount_Challenge/'
train = data_dir+'training.csv'  # path to training file
label = data_dir+'trainLabels.csv'  # path to label file of training data
test = data_dir+'test.csv'  # path to testing file

D = 2 ** 29  # number of weights use for each model, we have 32 of them
alpha = .1   # learning rate for sgd optimization


# function, generator definitions ############################################

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
hash_cols = [131,132,133,136,35,65,61,62,91,92,142,3,4,61,34,91,94,95]
hh=len(hash_cols)
hh=hh*(hh-1)/2+1
def data(path, label_path=None):
    for t, line in enumerate(open(path)):
        # initialize our generator
        
        #hash_cols = [61,62,91,92,142,3,4,34,35,61,64,65,91,94,95]
        if t == 0:
            # create a static x,
            # so we don't have to construct a new x for every instance
            
            x = [0] * (146+hh)
            if label_path:
                label = open(label_path)
                label.readline()  # we don't need the headers
            continue
        # parse x
        for m, feat in enumerate(line.rstrip().split(',')):
            if m == 0:
                ID = int(feat)
            elif m in [15,76,82,17,131,132,133,136,134,78,48,79,109,21,18,108,35,65,61,62,91,92,142,3,4,61,34,91,94,95,142,43,2,23,22,113,114,53,54,138,139, 96, 97, 98, 99,  100,  19,  29,  36, 37, 38, 39, 122, 110, 120, 121, 123, 124, 125, 59, 52, 50, 7, 6, 8, 9,  145, 122, 39, 38, 37, 36]: 
             
            #elif m in [23,22,113,114,53,54,138,139,70, 77, 96, 97, 98, 99, 107, 135, 100, 137, 132, 19, 16, 29, 28, 36, 37, 38, 39, 122, 144, 145, 47, 40, 110, 119, 60, 120, 121, 123, 124, 125, 59, 52, 50, 7, 6, 8, 9, 40, 144, 145, 122, 39, 38, 37, 36]:
                x[m] =-10
            else:
                # one-hot encode everything with hash trick
                # categorical: one-hotted
                # boolean: ONE-HOTTED
                # numerical: ONE-HOTTED!
                # note, the build in hash(), although fast is not stable,
                #       i.e., same value won't always have the same hash
                #       on different machines
                x[m] = abs(hash(str(m) + '_' + feat)) % D
        # parse y, if provided
        row=line.rstrip().split(',')
        
        t = 146
        for i in xrange(len(hash_cols)):
            for j in xrange(i+1,len(hash_cols)):
                t += 1
                x[t] = abs(hash(str(i)+'_'+str(j)+'_'+row[hash_cols[i]]+"_x_"+row[hash_cols[j]])) % D
        #print t  #t=145+hh
        #assert(false)
        
        
                
        if label_path:
            # use float() to prevent future type casting, [1:] to ignore id
            y = [float(y) for y in label.readline().split(',')[1:]][-1]
        yield (ID, x, y) if label_path else (ID, x)


# B. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     bounded logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)



def predict(x, w):
    wTx = 0.
    for i in x:  # do wTx
        if i <0:
            continue
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


def update(alpha, w, n, x, p, y):
    for i in x:
        if i <0:
            continue
        # alpha / sqrt(n) is the adaptive learning rate
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1.
        n[i] += abs(p - y)
        w[i] -= (p - y) * 1. * alpha / sqrt(n[i])
   



# training and testing #######################################################
start = datetime.now()

# a list for range(0, 33) - 13, no need to learn y14 since it is always 0


# initialize our model, all 32 of them, again ignoring y14
w = [0.] * D
n = [0.] * D

loss = 0.


losse=[0.466324, 0.379265, 0.335325, 0.308551, 0.287932, 0.272772, 0.261691, 0.251929, 0.243392, 0.236228, 0.229506, 0.223714, 0.218429, 0.213904, 0.209744, 0.205883, 0.202317]
cx=0
for ID, x, y in data(train, label):
    if True:
        p = predict(x, w)
      
        update(alpha, w, n, x, p, y)        
        loss += logloss(p, y)

    if ID % 100000 == 0:
        if cx>=17:
            continue
        print('%s encountered: %d logloss33: %f  total_logloss: %f '  % (
            datetime.now(), ID, (loss*1.)/ID, (loss*1./ID+losse[cx])/33))
        cx+=1

with open(data_dir+'./submissiontk33-33.csv', 'w') as outfile:
    outfile.write('id_label,pred\n')
    
    for ID, x in data(test):
        if True:
            
            p = predict(x, w)
        
            outfile.write('%s_y%d,%s\n' % (ID, 33, str(p)))
            
#pickle.dump( w, open( "tk16-weights.p", "wb" ) )
print('Done, elapsed time: %s' % str(datetime.now() - start))