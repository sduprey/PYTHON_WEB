'''
Created on 3 Jul 2015

@author: sduprey
'''

class my_online_learner:

    def __init__(self,train="/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_training_matrix.csv",label = '/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_labelling_vector.csv',test= "/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_testing_matrix.csv",alpha= .1,output_file='/home/sduprey/My_Data/My_Cdiscount_Challenge/submission.csv'):
        self.train=train
        self.label=label
        self.test=test
        self.alpha=alpha
        self.output_file=output_file
        
        
    def data(selfself, path, label_path=None):