'''
Created on 7 Jul 2015

@author: sduprey
'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier

class rf_classifier_full_hashed :
    
    def __init__(self,n_estimators=200, n_jobs=16, min_samples_leaf = 10,random_state=1,bootstrap=False,criterion='entropy',min_samples_split=5,verbose=1,nb_categories=5789) :
        self.n_estimators=n_estimators
        self.n_jobs=n_jobs
        self.min_samples_leaf=min_samples_leaf
        self.random_state=random_state
        self.bootstrap=bootstrap
        self.criterion=criterion  
        self.min_samples_split=min_samples_split
        self.verbose=verbose
        self.nb_categories=nb_categories
    
    def train_predict(self,X_train,y_train,X_test):
        rf = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=self.n_jobs, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state, bootstrap=self.bootstrap, criterion=self.criterion, min_samples_split=self.min_samples_split, verbose=self.verbose)
        print('Training random forest model')
        fitted_model = rf.fit(X_train, y_train)
        print('Predicting on the test samlpe using the random forest model')
        ypred = fitted_model.predict_proba(X_test)
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
    