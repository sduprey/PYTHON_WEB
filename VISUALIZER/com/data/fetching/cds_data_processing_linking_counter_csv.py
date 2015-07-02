#!/usr/bin/python
# coding: utf-8
'''
Created on 17 March 2015

@author: sduprey
'''
import psycopg2
import numpy as np
from com.data.fetching.utility import save_histogram_as_csv_file

def main():
    #Define our connection string
    csv_saving_path = '/home/sduprey/My_Data/My_Kriter_Data/My_Kriter_Results/'
    current_parameter_checked = 'cds_linking'
    magasin_to_display = ''

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

    save_histogram_as_csv_file(current_parameter_checked, magasin_to_display,X,csv_saving_path)
        
if __name__ == "__main__":
    main()