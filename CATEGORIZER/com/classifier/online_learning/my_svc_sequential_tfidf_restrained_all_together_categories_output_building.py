'''
Created on 10 Jul 2015

@author: sduprey
'''
import numpy as np

def load_previously_saved_predictions(filename):
    loader = np.load(filename)
    return loader['data']


if __name__ == "__main__":

    filename_category_model = "/home/sduprey/My_Data/My_Cdiscount_Challenge/svc_sequential_whole_tfidf_restrained_data_all_together_submission.csv.npz"
    print("Saving model to our file : " + filename_category_model)
    
    predictions = load_previously_saved_predictions(filename_category_model)
    print(type(predictions))
    print(predictions.shape[0])
    print(predictions.shape[1])
    
    for i in range(predictions.shape[0]) :
        print(sum(predictions[i,:]))


    
    
    
    
    