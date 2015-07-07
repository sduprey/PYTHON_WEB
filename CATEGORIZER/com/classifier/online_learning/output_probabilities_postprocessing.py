'''
Created on 7 Jul 2015

@author: sduprey
'''
import csv

if __name__ == "__main__":
    my_output_file = "/home/sduprey/My_Data/My_Cdiscount_Challenge/sequential_all_together_final_submission.csv"
    with open(my_output_file, 'rb') as csvfile :
        #spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        outputreader = csv.reader(csvfile, delimiter=";")
        for rowid, rowvalues in enumerate(outputreader) :
            print('Dealing with row id '+ str(rowid))
            print('List of probabilities of length : ' + str(len(rowvalues)))