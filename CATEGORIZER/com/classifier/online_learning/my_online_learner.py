'''
Created on 3 Jul 2015

@author: sduprey
'''

from datetime import datetime
from math import log, exp, sqrt,factorial

class my_online_learner:
    def __init__(self,train="/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_training_matrix.csv",label = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_data_training_output_vector.csv',test="/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_testing_matrix.csv", D=1048576 ,alpha= .1 ,output_file="/home/sduprey/My_Data/My_Cdiscount_Challenge/submission.csv" ,nb_categories=5789 ,category_number=1473):
        self.train=train
        self.label=label
        self.test=test
        self.alpha=alpha
        self.output_file=output_file
        self.nb_categories=nb_categories 
        self.category_number=category_number
        self.D=D
        
    def data(self, path, label_path=None):
        if label_path:
            label=open(label_path)
            
        indices = []
        rowDatas = []
        rowNumber = 0
        for t, line in enumerate(open(path)):
            if t%3 == 0 :
                print('Row number '+line)
                rowNumber = int(line.rstrip()) 
            if t%3 == 1 :
                print("My indices")
                indices =  [int(l) for l in line.rstrip().split(',')] 
            if t%3 == 2 :
                print('My row data '+line)
                rowDatas =  [float(l) for l in line.rstrip().split(',')] 
                if label_path:
                # use float() to prevent future type casting, [1:] to ignore id
                    y = int(label.readline().rstrip())
                yield(rowNumber, indices, rowDatas, y)  if label_path else (rowNumber, indices, rowDatas)

    def predict(self, indices, rowdata, w):
        wTx = 0.
        data_counter = 0
        for indice in indices :
            wTx += w[indice]*rowdata[data_counter]
            data_counter=data_counter+1
        return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


    def train_predict(self):
        start = datetime.now()
        # a model for each of our 5789 categories
        K = [k for k in range(self.nb_categories)]

        # initialize our model, all 5789 nb_categories of them
        w = [[0.] * (self.D+self.nb_categories) for k in K]
        n = [[0.] * (self.D+self.nb_categories) for k in K]

        loss = 0.
        loss2 = 0.
        loss_y14 = log(1. - 10**-15)

        for rowNumber, indices, rowDatas, label in self.data(self.train, self.label):
            print('Dealing with row number '+str(rowNumber))
            predictions=[0.]*self.nb_categories
            P=[]
            
            # dealing with the category K
            for k in K:
                p = self.predict(indices, rowDatas, w[k])
                P.append(p)
                predictions[k]=p
                good_answer = label == k
                loss += self.logloss(p, good_answer) 
                
        for ID, x, y in self.data(self.train, self.label):

    # get predictions and train on all labels
            P=[]
            for k in K:
                p = self.predict(x, w[k])
                P.append(p)
       # update(alpha, w[k], n[k], x, p, y[k])
                if k<13:
                    x[146+self.hh+k]=p
                else:
                    x[145+self.hh+k]=p
                loss += self.logloss(p, y[k]) 
            for k,p in zip(K,P):
                p2 = self.predict2(x, w[k])
                self.update(self.alpha, w[k], n[k], x, p2, y[k])
                self.update2(self.alpha, w[k], n[k], x, p2, y[k])
                loss2 += self.logloss(p2, y[k])  
            loss += loss_y14  # the loss of y14, logloss is never zero
            loss2 += loss_y14
    # print out progress, so that we know everything is working
            if ID % 100 == 0:
        
                print('%s encountered: %d current logloss: %f  logloss2: %f' % (
                    datetime.now(), ID, (loss/33.)/ID,(loss2/33.)/ID))
   
           
        with open(self.output_file, 'w') as outfile:
            outfile.write('id_label,pred\n')
    
            for ID, x in self.data(self.test):
                for k in K:
                    p = self.predict(x, w[k])
                    if k<13:
                        x[146+self.hh+k]=p
                    else:
                        x[145+self.hh+k]=p
                for k in K:
                    p = self.predict2(x, w[k])
                    outfile.write('%s_y%d,%s\n' % (ID, k+1, str(p)))
                    if k == 12 and self.predict_y14:
                        outfile.write('%s_y14,0.0\n' % ID)

        print('Done, elapsed time: %s' % str(datetime.now() - start))

if __name__ == "__main__":
    best_online = my_online_learner(train="/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_training_matrix.csv",label = '/home/sduprey/My_Data/My_Cdiscount_Challenge/hashed_data_training_output_vector.csv',test="/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_testing_matrix.csv", D=1048576 , alpha= .1, output_file="/home/sduprey/My_Data/My_Cdiscount_Challenge/submission.csv", nb_categories=5789, category_number=1473)
    best_online.train_predict()
    
    
    