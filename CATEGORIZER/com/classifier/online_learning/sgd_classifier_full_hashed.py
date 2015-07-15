'''
Created on 7 Jul 2015

@author: sduprey
'''

import numpy as np
from sklearn.linear_model import SGDClassifier


class sgd_classifier_full_hashed :
    def __init__(self, loss='log', alpha=0.000001, n_iter=100, nb_categories=5789) :
        self.nb_categories=nb_categories
        self.loss=loss
        self.alpha=alpha
        self.n_iter=n_iter
    
    def train_predict(self,X_train,y_train,X_test):        
        clf=SGDClassifier(loss=self.loss,alpha=self.alpha, n_iter=self.n_iter)
        print('Training stochastic gradient descent model')
        fitted_model = clf.fit(X_train, y_train)
        print('Predicting on the test samlpe using stochastic gradient descent  model')
        ypred = fitted_model.predict_proba(X_test)
        ypredtrain = fitted_model.predict_proba(X_train)
        return ypred, ypredtrain
    
    def train_predict_all_labels(self,X_train,y_train,X_test):
        xgb_predicttest=[]  
        xgb_predicttrain=[]         
        for i in range(self.nb_categories) :
            y_i=np.zeros(y_train.shape, dtype=np.int)
            y_i[y_train == (i+1)] = 1
            y_i[y_train != (i+1)] = 0
            print('Dealing with category : '+str(i+1))
            print('Category size : '+str(sum(y_i)))            
            #predicting the category i using our xgb model
            predicted, predictedtrain = self.train_predict(X_train, y_i, X_test)
            xgb_predicttest.append(predicted)
            xgb_predicttrain.append(predictedtrain)
        return np.column_stack(xgb_predicttest),  np.column_stack(xgb_predicttrain)
    