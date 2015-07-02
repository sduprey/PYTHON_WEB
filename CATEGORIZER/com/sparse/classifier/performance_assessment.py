'''
Created on 26 May 2015

@author: sduprey
'''
import psycopg2
from sklearn import metrics

algos = [
'MULTINOMIALNB',
'BERNOULLINB',
'RIDGECLASSIFIER',
'PASSIVEAGGRESSIVE',
'KNN',
'LINEARSVCL1',
'LINEARSVCL2',
'SGDL1',
'SGDL2',
'NEARESTCENTROID',
'PERCEPTRON',
'LINEARSVCL1FS',
'LinearSVCCLASS']

conn_string = "host='localhost' dbname='CATEGORIZERDB' user='postgres' password='mogette'"
# print the connection string we will use to connect
 
# get a connection, if a connect cannot be made an exception will be raised here
conn = psycopg2.connect(conn_string)
conn.autocommit = True
# conn.cursor will return a cursor object, you can use this cursor to perform queries
cursor = conn.cursor()



def process_algo(name):
    sql_request ='select identifiant_produit, real_category, prediction from '+name
    print sql_request
    cursor.execute(sql_request); 
    # retrieve the records from the database
    results_data = cursor.fetchall()

    ytest = [item[1] for item in results_data];
    pred  = [item[2] for item in results_data];
    benchmark(name, ytest, pred)
    
def benchmark(name,y_test, pred):
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
#    print("classification report:")
#    print(metrics.classification_report(y_test, pred))
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

# assessing each algorithm one after the other
for algo in algos :
    process_algo(algo)



