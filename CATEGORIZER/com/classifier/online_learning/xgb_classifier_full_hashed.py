'''
Created on 7 Jul 2015

@author: sduprey
'''

import xgboost as xgb
import numpy as np

class xgb_classifier_full_hashed :
    
    def __init__(self,eta,min_child_weight,depth,num_round,threads=8,exist_prediction=0,exist_num_round=20,nb_categories=5789):
        self.eta=eta
        self.min_child_weight=min_child_weight
        self.depth=depth
        self.num_round=num_round
        self.exist_prediction=exist_prediction
        self.exist_num_round=exist_num_round  
        self.threads=threads
        self.nb_categories=nb_categories
    
    
    def train_predict(self,X_train,y_train,X_test):
        xgmat_train = xgb.DMatrix(X_train, label=y_train,missing=-999)
        #test_size = X_test.shape[0]
        param = {}
        param['objective'] = 'binary:logistic'

        param['bst:eta'] = self.eta
        param['colsample_bytree']=1
        param['min_child_weight']=self.min_child_weight
        param['bst:max_depth'] = self.depth
        param['eval_metric'] = 'logloss'
        param['silent'] = 1
        param['nthread'] = self.threads
        plst = list(param.items())

        watchlist = [ (xgmat_train,'train') ]
        num_round = self.num_round
    
        print('Training xgd model')
        bst = xgb.train( plst, xgmat_train, num_round, watchlist )
        xgmat_test = xgb.DMatrix(X_test,missing=-999)
    
        if self.exist_prediction:
        # train xgb with existing predictions
        # see more at https://github.com/tqchen/xgboost/blob/master/demo/guide-python/boost_from_prediction.py
       
            tmp_train = bst.predict(xgmat_train, output_margin=True)
            tmp_test = bst.predict(xgmat_test, output_margin=True)
            xgmat_train.set_base_margin(tmp_train)
            xgmat_test.set_base_margin(tmp_test)
            bst = xgb.train(param, xgmat_train, self.exist_num_round, watchlist )

        print('Predicting on the test samlpe using the xgd model')
        ypred = bst.predict(xgmat_test)
        ypretrain =  bst.predict(xgmat_train)
        return ypred, ypretrain

    def train_predict_all_specific_labels(self,thread_name, my_category_chunk ,X_train ,y_train ,X_test):
        xgb_predict=[]        
        for i in my_category_chunk :
            y_i=np.zeros(y_train.shape, dtype=np.int)
            y_i[y_train == (i+1)] = 1
            y_i[y_train != (i+1)] = 0
            print(thread_name+' : Dealing with category : '+str(i+1))
            print(thread_name+' :Category size : '+str(sum(y_i)))            
            #predicting the category i using our xgb model
            predicted = self.train_predict(X_train, y_i, X_test)
            xgb_predict.append(predicted)
        return np.column_stack(xgb_predict)

    
    def train_predict_all_labels(self,X_train,y_train,X_test):
        xgb_predict_test=[]   
        xgb_predict_train=[]     
        for i in range(self.nb_categories) :
            y_i=np.zeros(y_train.shape, dtype=np.int)
            y_i[y_train == (i+1)] = 1
            y_i[y_train != (i+1)] = 0
            print('Dealing with category : '+str(i+1))
            print('Category size : '+str(sum(y_i)))            
            #predicting the category i using our xgb model
            predicted, predicted_train = self.train_predict(X_train, y_i, X_test)
            xgb_predict_test.append(predicted)
            xgb_predict_train.append(predicted_train)
        return np.column_stack(xgb_predict_test), np.column_stack(xgb_predict_train)
    
    
        
    def train_predict_one_shot(self,X_train,y_train,X_test):     
            
        xgmat_train = xgb.DMatrix(X_train, label=y_train,missing=-999)
        #test_size = X_test.shape[0]
        param = {}
        param['objective'] = 'binary:logistic'

        param['bst:eta'] = self.eta
        param['colsample_bytree']=1
        param['min_child_weight']=self.min_child_weight
        param['bst:max_depth'] = self.depth
        param['eval_metric'] = 'logloss'
        param['silent'] = 1
        param['nthread'] = self.threads
        plst = list(param.items())

        watchlist = [ (xgmat_train,'train') ]
        num_round = self.num_round
    
        print('Training xgd model for all categories')
        bst = xgb.train( plst, xgmat_train, num_round, watchlist )
        xgmat_test = xgb.DMatrix(X_test,missing=-999)
    

        print('Predicting on the test samlpe using the xgd model')
        ypred = bst.predict(xgmat_test)
            
            
        return ypred
    