'''
Created on 9 Jul 2015

@author: sduprey
'''

import threading
from xgb_classifier_full_hashed import xgb_classifier_full_hashed
import numpy as np

class xgdClassifierThread (threading.Thread):
    def __init__(self, threadID, name, Xtrain, ytrain, Xtest, my_categories_chunk):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.my_categories_chunk = my_categories_chunk
    
    def run(self):
        print "Starting " + self.name
        stepping_through_categories(self.name, self.Xtrain, self.ytrain, self.Xtest, self.my_categories_chunk)
        print "Exiting " + self.name

def save_model(filename,vector):
    np.savez(filename,data =vector)

def stepping_through_categories(threadName, Xtrain, ytrain, Xtest, my_category_chunk):
    print "%s beginning to deal with its categories %s" % (threadName, str(my_category_chunk))

    print(threadName + 'Testing sample size')
    print(Xtest.shape[0])
    print(Xtest.shape[1])
    
    print(threadName + 'Training sample size')
    print(Xtest.shape[0])
    print(Xtest.shape[1])

    print(threadName + 'Output sample size')
    print(ytrain.shape[0])

    print(threadName +' Training and predicting xgb classifier for each category')
    xgb_clf=xgb_classifier_full_hashed(eta=0.3,min_child_weight=6,depth=100,num_round=20,threads=30,exist_prediction=False,exist_num_round=20, nb_categories=5789) 
    my_thread_X_test_xgb_pred = xgb_clf.train_predict_all_specific_labels(threadName, my_category_chunk, Xtrain, ytrain,Xtest)
    
    filename_category_model = "/home/sduprey/My_Data/My_Cdiscount_Challenge/xgb_parallel_all_together_submission"+threadName+".csv"
    print(threadName +" : Saving model to our file : " + filename_category_model)
    save_model(filename_category_model, my_thread_X_test_xgb_pred)
    
    

#def print_time(threadName, delay, counter):
#    while counter:
#        if exitFlag:
#            thread.exit()
#        time.sleep(delay)
#        print "%s: %s" % (threadName, time.ctime(time.time()))
#        counter -= 1
