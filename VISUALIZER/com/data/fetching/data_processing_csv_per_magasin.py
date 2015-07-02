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
    conn_string = "host='localhost' dbname='KRITERDB' user='postgres' password='mogette'"
    # print the connection string we will use to connect
    print "Connecting to database\n    ->%s" % (conn_string)
 
    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)
 
    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor()

    my_distinct_magasin_request = "select distinct magasin from CATALOG"

    cursor.execute(my_distinct_magasin_request); 
        # retrieve the records from the database
    datas = cursor.fetchall()
    my_magasins = [item[0] for item in datas];
    current_parameter_checked="";
    #my_magasins = ["Musique","Librairie"];
    for magasin_to_loop in my_magasins:
        magasin_to_display=magasin_to_loop.replace (" ", "_")
        my_brand_request = "select distinct nb_distinct_brand, count(*) from CATALOG where nb_distinct_brand is not null and magasin=(%s) group by nb_distinct_brand order by nb_distinct_brand asc"
        current_parameter_checked="distinct_brand" 
        print "Dealing with :"+current_parameter_checked
        print "Executing the following request to fetch data for  magasins : "+magasin_to_loop + my_brand_request
        # fetching data to display for magasin Musique
        cursor.execute(my_brand_request,(magasin_to_loop,)); 
        # retrieve the records from the database
        numerical_data = cursor.fetchall()

        X= np.asanyarray(numerical_data);
        #y= np.asanyarray(y);
        print type(X)
        print X.shape
        print X.dtype
        print X.size
        save_histogram_as_csv_file(current_parameter_checked, magasin_to_display,X,results_saving_path)
        

        my_magasin_request = "select distinct nb_distinct_magasin, count(*) from CATALOG where nb_distinct_magasin is not null and magasin=(%s) group by nb_distinct_magasin order by nb_distinct_magasin asc"
        current_parameter_checked="distinct_magasin" 
        print "Dealing with :"+current_parameter_checked
        print "Executing the following request to fetch data for  magasins : " +magasin_to_loop+ my_magasin_request
    
        # fetching data to display for magasin Musique
        cursor.execute(my_magasin_request,(magasin_to_loop,)); 
        # retrieve the records from the database
        numerical_data = cursor.fetchall()

        X= np.asanyarray(numerical_data);
        #y= np.asanyarray(y);
        print type(X)
        print X.shape
        save_histogram_as_csv_file(current_parameter_checked, magasin_to_display,X,results_saving_path)
        my_state_request = "select distinct nb_distinct_state, count(*) from CATALOG where nb_distinct_state is not null and magasin=(%s) group by nb_distinct_state order by nb_distinct_state asc"
        current_parameter_checked="distinct_state" 
        print "Dealing with :"+current_parameter_checked
        print "Executing the following request to fetch data for  magasins : "+magasin_to_loop + my_state_request
    
        # fetching data to display for magasin Musique
        cursor.execute(my_state_request,(magasin_to_loop,)); 
        # retrieve the records from the database
        numerical_data = cursor.fetchall()

        X= np.asanyarray(numerical_data);
        #y= np.asanyarray(y);
        print type(X)
        print X.shape
        save_histogram_as_csv_file(current_parameter_checked, magasin_to_display,X,results_saving_path)

        my_cat4_request = "select distinct nb_distinct_cat4, count(*) from CATALOG where nb_distinct_cat4 is not null and magasin=(%s) group by nb_distinct_cat4 order by nb_distinct_cat4 asc"
        current_parameter_checked="distinct_category_level_4" 
        print "Dealing with :"+current_parameter_checked
        print "Executing the following request to fetch data for  magasins : "+magasin_to_loop + my_cat4_request
    
        # fetching data to display for magasin Musique
        cursor.execute(my_cat4_request,(magasin_to_loop,)); 
        # retrieve the records from the database
        numerical_data = cursor.fetchall()

        X= np.asanyarray(numerical_data);
        #y= np.asanyarray(y);
        print type(X)
        print X.shape
        save_histogram_as_csv_file(current_parameter_checked, magasin_to_display,X,results_saving_path)
        
if __name__ == "__main__":
    main()