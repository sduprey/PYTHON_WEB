'''
Created on 25 Jun 2015

@author: sduprey
'''


my_full_hashed_training_matrix_file = '/home/sduprey/My_Data/My_Cdiscount_Challenge/full_hashed_training_matrix.csv'
indices = []
rowDatas = []
rowNumber = 0
for t, line in enumerate(open(my_full_hashed_training_matrix_file)):
    if t%3 == 0 :
        print('Row number '+line)
        rowNumber = int(line.rstrip()) 
#        for m, feat in enumerate(line.rstrip().split(',')):
#            print(feat)   
    if t%3 == 1 :
        print("My indices")
        indices =  [int(l) for l in line.rstrip().split(',')] 
#        for m, feat in enumerate(line.rstrip().split(',')):
#            print(feat)
    if t%3 == 2 :
        print('My row data '+line)
        rowDatas =  [float(l) for l in line.rstrip().split(',')] 
        print(rowDatas)
       
    
    
