'''
Created on 7 Jul 2015

@author: sduprey
'''

import numpy as np
from sklearn.svm import LinearSVC

class svc_classifier_full_hashed :
    def __init__(self, nb_categories=5789) :
        self.nb_categories=nb_categories
    
    def train_predict(self,X_train,y_train,X_test):        
        clf=LinearSVC(C=0.17)
        print('Training linear support vector machine model')
        fitted_model = clf.fit(X_train, y_train)
        print('Predicting on the test samlpe using linear support vector machine model')
        ypred = fitted_model.predict(X_test)
        return ypred
    
    def train_predict_all_labels(self,X_train,y_train,X_test):
        xgb_predict=[]        
        for i in range(self.nb_categories) :
            y_i=np.zeros(y_train.shape, dtype=np.int)
            y_i[y_train == (i+1)] = 1
            y_i[y_train != (i+1)] = 0
            print('Dealing with category : '+str(i+1))
            print('Category size : '+str(sum(y_i)))            
            #predicting the category i using our xgb model
            predicted = self.train_predict(X_train, y_i, X_test)
            xgb_predict.append(predicted)
        return np.column_stack(xgb_predict)
    