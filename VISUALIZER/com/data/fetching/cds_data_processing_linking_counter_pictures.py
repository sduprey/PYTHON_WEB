#!/usr/bin/python
# coding: utf-8
'''
Created on 17 March 2015

@author: sduprey
'''
import psycopg2
import numpy as np
import matplotlib.pyplot as plt

def main():
    #Define our connection string
    pictures_saving_path = '/home/sduprey/My_Data/My_Kriter_Data/My_Kriter_Pictures/'


    conn_string = "host='localhost' dbname='KRITERDB' user='postgres' password='mogette'"
    # print the connection string we will use to connect
    print "Connecting to database\n    ->%s" % (conn_string)
 
    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)
 
    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor()
  
    # execute our Query
    # X = np.asarray(predictors_list);
    my_linking_counter_request = "select distinct counter, count(*) from CDS_LINKING_SIMILAR_PRODUCTS group by counter order by counter asc"
    print "Executing the following request to fetch data for all magasins : " + my_linking_counter_request
    
    # fetching data to display for magasin Musique
    cursor.execute(my_linking_counter_request); 
    # retrieve the records from the database
    numerical_data = cursor.fetchall()
    X= np.asanyarray(numerical_data);
    #y= np.asanyarray(y);
    print type(X)
    print X.shape
    plt.plot(X)
    #plt.hist(X)
    plt.title(unicode("Links Number Histogram",'utf-8'))
    plt.xlabel("Number of links")
    plt.ylabel("Number of Skus")
    plt.show();
    plt.savefig(unicode(pictures_saving_path+"cds_nblinks_per_nbskus.png",'utf-8'))
        
if __name__ == "__main__":
    main()