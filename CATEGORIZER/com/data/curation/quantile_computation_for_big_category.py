'''
Created on 28 May 2015

@author: sduprey
'''

import psycopg2
import numpy as np

print("Loading the big categories with a count greater than 20000 ")
sql_big_category_requesting = 'select categorie_3 from training_category_following where count >20000 order by count desc'
conn_string = "host='localhost' dbname='CATEGORIZERDB' user='postgres' password='mogette'"
# print the connection string we will use to connect
 
# get a connection, if a connect cannot be made an exception will be raised here
conn = psycopg2.connect(conn_string)
conn.autocommit = True
# conn.cursor will return a cursor object, you can use this cursor to perform queries
cursor = conn.cursor()
cursor.execute(sql_big_category_requesting); 
 # retrieve the records from the database
items = cursor.fetchall()
fetched_categories = [item[0] for item in items]
def compute_quantiles(category):
    sql_big_category_length_requesting = 'select length(libelle), length(description) from training_data where categorie_3 =(%s)'
    print('Requesting category for '+category)
    cursor.execute(sql_big_category_length_requesting,(category,));
    numerical_data = cursor.fetchall()
    X= np.asanyarray(numerical_data);
    p_libelle = np.percentile(X[:,0], 90) # return 50th percentile, e.g median.
    p_description = np.percentile(X[:,1], 90) # return 50th percentile, e.g median.
    print('Category :'+category)
    print('Libelle 90% quantile :'+str(p_libelle))
    print('Description 90% quantile :'+str(p_description))

[compute_quantiles(cat) for cat in fetched_categories]



#training_identifiant_produit_list = [item[0] for item in fetched_training_data];
#training_outputs  = [item[1] for item in fetched_training_data];
#training_documents  = [item[2] + ' '+item[3] for item in fetched_training_data];