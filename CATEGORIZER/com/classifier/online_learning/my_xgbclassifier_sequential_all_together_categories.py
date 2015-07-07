'''
Created on 7 Jul 2015

@author: sduprey
'''

import csv
import numpy as np
from scipy import sparse
from xgb_classifier_full_hashed import xgb_classifier_full_hashed

class my_xgb_classifier_all_together_category:
    def __init__(self, D=1048576 ,nb_categories=5789):
        self.nb_categories=nb_categories
        self.D=D
    
    def load_csv_output_vector(self, filename): 
        with open(filename, 'rb') as csvfile:
            output_vector_reader = csv.reader(csvfile, delimiter=';')  
            y=[]
            for row in output_vector_reader:
                y.append(int(row[0].rstrip())) 
            return y   
        
    def load_sparse_csr(self, filename):
        loader = np.load(filename)
        return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        
    def save_model(self, filename,vector):
        np.savez(filename,data =vector)
    
    def train_predict_all_labels(self,Xtrain, ytrain,Xtest):
        #xgb_clf=xgb_classifier_full_hashed(eta=0.3,min_child_weight=6,depth=100,num_round=20,threads=16,exist_prediction=True,exist_num_round=20, nb_categories=self.nb_categories)
        # we here ignore the real output : exist_prediction = False
        xgb_clf=xgb_classifier_full_hashed(eta=0.3,min_child_weight=6,depth=100,num_round=20,threads=16,exist_prediction=False,exist_num_round=20, nb_categories=self.nb_categories) 
        X_test_xgb_pred = xgb_clf.train_predict_all_labels(Xtrain, ytrain,Xtest)
        return X_test_xgb_pred
        
 
if __name__ == "__main__":
#    nb_categories = 5789
#    D=1048576
    xgb_all_categories = my_xgb_classifier_all_together_category(D=5789, nb_categories=1048576)
    
    my_saving_training_matrix_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_training_matrix.bin.npz'
    Xtrain = xgb_all_categories.load_sparse_csr(my_saving_training_matrix_file)
    print('Training sample size')
    print(Xtrain.shape[0])
    print(Xtrain.shape[1])

    my_saving_training_outputvector_file_path = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_data_training_output_vector.csv'
    ytrain = xgb_all_categories.load_csv_output_vector(my_saving_training_outputvector_file_path)
    ytrain = np.asarray(ytrain)
    print('Training sample label size')
    print(ytrain.shape[0])

    my_saving_testing_matrix_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_testing_matrix.bin.npz'
    Xtest = xgb_all_categories.load_sparse_csr(my_saving_testing_matrix_file)
    print('Testing sample size')
    print(Xtest.shape[0])
    print(Xtest.shape[1])
      
    # getting the weighted model for our category
    print('Training and predicting xgb classifier for each category')
    my_xgb_predictions = xgb_all_categories.train_predict_all_labels(Xtrain, ytrain, Xtest)
    
    
    
    filename_category_model = "/home/sduprey/My_Data/My_Cdiscount_Challenge/xgb_sequential_all_together_submission.csv"
    print("Saving model to our file : " + filename_category_model)
    xgb_all_categories.save_model(filename_category_model, my_xgb_predictions)
    
    # saving the model for the category number
    
    
    
    
    
    