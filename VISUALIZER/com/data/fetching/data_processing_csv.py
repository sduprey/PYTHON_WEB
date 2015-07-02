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
    results_saving_path = '/home/sduprey/My_Data/My_Kriter_Data/My_Kriter_Results/'
    magasin_to_display = ''

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
    my_brand_request = "select distinct nb_distinct_brand, count(*) from CATALOG where nb_distinct_brand is not null group by nb_distinct_brand order by nb_distinct_brand  asc"

    print "Executing the following request to fetch data for all magasins : " + my_brand_request
    current_parameter_checked="distinct_brand" 
    # fetching data to display for magasin Musique
    cursor.execute(my_brand_request); 
    # retrieve the records from the database
    numerical_data = cursor.fetchall()
    X= np.asanyarray(numerical_data);
    #y= np.asanyarray(y);
    print type(X)
    print X.shape
    save_histogram_as_csv_file(current_parameter_checked, magasin_to_display,X,results_saving_path)


    my_magasin_request = "select distinct nb_distinct_magasin, count(*) from CATALOG where nb_distinct_magasin is not null group by nb_distinct_magasin order by nb_distinct_magasin  asc"
    print "Executing the following request to fetch data for  magasins : " + my_magasin_request
    current_parameter_checked="distinct_magasin" 
    # fetching data to display for magasin Musique
    cursor.execute(my_magasin_request); 
    # retrieve the records from the database
    numerical_data = cursor.fetchall()

    X= np.asanyarray(numerical_data);
    #y= np.asanyarray(y);
    print type(X)
    print X.shape
    save_histogram_as_csv_file(current_parameter_checked, magasin_to_display,X,results_saving_path)

    my_state_request = "select distinct nb_distinct_state, count(*) from CATALOG where nb_distinct_state is not null group by nb_distinct_state order by nb_distinct_state  asc"
    print "Executing the following request to fetch data for  magasins : " + my_state_request
    current_parameter_checked="distinct_state" 
    # fetching data to display for magasin Musique
    cursor.execute(my_state_request); 
     # retrieve the records from the database
    numerical_data = cursor.fetchall()

    X= np.asanyarray(numerical_data);
        #y= np.asanyarray(y);
    print type(X)
    print X.shape
    save_histogram_as_csv_file(current_parameter_checked, magasin_to_display,X,results_saving_path)
    
    my_cat4_request = "select distinct nb_distinct_cat4, count(*) from CATALOG where nb_distinct_cat4 is not null group by nb_distinct_cat4 order by nb_distinct_cat4  asc"
    print "Executing the following request to fetch data for  magasins : "+ my_cat4_request
    current_parameter_checked="distinct_cat4" 
        # fetching data to display for magasin Musique
    cursor.execute(my_cat4_request); 
        # retrieve the records from the database
    numerical_data = cursor.fetchall()

    X= np.asanyarray(numerical_data);
        #y= np.asanyarray(y);
    print type(X)
    print X.shape
    save_histogram_as_csv_file(current_parameter_checked, magasin_to_display,X,results_saving_path)

if __name__ == "__main__":
    main()