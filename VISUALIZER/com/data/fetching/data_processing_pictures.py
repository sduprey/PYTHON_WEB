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
    my_full_request = "select nb_distinct_cat5, nb_distinct_cat4, nb_distinct_brand_without_default, nb_distinct_vendor, nb_distinct_magasin, nb_distinct_state from CATALOG where magasin=(%s)"
    dataNames=["Category5","Category4","Brand","Vendor","Magasin","State"];
    print "Data fetched : "+ ', '.join(dataNames)

    
    # execute our Query
    # X = np.asarray(predictors_list);
    my_brand_request = "select nb_distinct_brand from CATALOG where nb_distinct_brand is not null"
    print "Executing the following request to fetch data for all magasins : " + my_brand_request
    
    # fetching data to display for magasin Musique
    cursor.execute(my_brand_request); 
    # retrieve the records from the database
    numerical_data = cursor.fetchall()
    X= np.asanyarray(numerical_data);
    #y= np.asanyarray(y);
    print type(X)
    print X.shape
    plt.hist(X)
    plt.title(unicode("nb_distinct_brand Histogram",'utf-8'))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(unicode(pictures_saving_path+"nb_distinct_brand.png",'utf-8'))
        
    my_magasin_request = "select nb_distinct_magasin from CATALOG where nb_distinct_magasin is not null"
    print "Executing the following request to fetch data for  magasins : " + my_magasin_request
    
    # fetching data to display for magasin Musique
    cursor.execute(my_magasin_request); 
    # retrieve the records from the database
    numerical_data = cursor.fetchall()

    X= np.asanyarray(numerical_data);
    #y= np.asanyarray(y);
    print type(X)
    print X.shape
    plt.hist(X)
    plt.title(unicode("nb_distinct_magasin Histogram",'utf-8'))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(unicode(pictures_saving_path+"nb_distinct_magasin_"+".png",'utf-8'))
        
    my_state_request = "select nb_distinct_state from CATALOG where nb_distinct_state is not null"
    print "Executing the following request to fetch data for  magasins : " + my_state_request
    
    # fetching data to display for magasin Musique
    cursor.execute(my_state_request); 
     # retrieve the records from the database
    numerical_data = cursor.fetchall()

    X= np.asanyarray(numerical_data);
        #y= np.asanyarray(y);
    print type(X)
    print X.shape
    plt.hist(X)
    plt.title(unicode('nb_distinct_state_'+'.png','utf-8'))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(unicode(pictures_saving_path+"nb_distinct_state_"".png",'utf-8'))
        #print type(y)
        #print y.shape
        
    my_cat4_request = "select nb_distinct_cat4 from CATALOG where nb_distinct_cat4 is not null"
    print "Executing the following request to fetch data for  magasins : "+ my_cat4_request
    
        # fetching data to display for magasin Musique
    cursor.execute(my_cat4_request); 
        # retrieve the records from the database
    numerical_data = cursor.fetchall()

    X= np.asanyarray(numerical_data);
        #y= np.asanyarray(y);
    print type(X)
    print X.shape
    plt.hist(X)
    plt.title(unicode("nb_distinct_cat4 Histogram",'utf-8'))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(unicode(pictures_saving_path+"nb_distinct_cat4_"+".png",'utf-8'))
        
    my_cat5_request = "select nb_distinct_cat5 from CATALOG where nb_distinct_cat5 is not null"
    print "Executing the following request to fetch data for  magasins : " + my_cat5_request
    
        # fetching data to display for magasin Musique
    cursor.execute(my_cat5_request); 
        # retrieve the records from the database
    numerical_data = cursor.fetchall()

    X= np.asanyarray(numerical_data);
        #y= np.asanyarray(y);
    print type(X)
    print X.shape
    plt.hist(X)
    plt.title(unicode("nb_distinct_cat5 Histogram",'utf-8'))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(unicode(pictures_saving_path+"nb_distinct_cat5_"+".png",'utf-8'))
        #print type(y)
        #print y.shape

if __name__ == "__main__":
    main()