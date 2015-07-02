#!/usr/bin/python
# coding: utf-8
'''
Created on 26 Jan 2015

@author: sduprey
'''
import pprint
import psycopg2
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.qda import QDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import sys

import numpy as np
import pandas as pd


def main():
    #Define our connection string
    conn_string = "host='localhost' dbname='CRAWL4J' user='postgres' password='mogette'"
    # print the connection string we will use to connect
    print "Connecting to database\n    ->%s" % (conn_string)
 
    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)
    
    # fetching training data from Cdiscount-maison
    cdiscount_maison_request = "select url, whole_text, title, h1, short_description, status_code, depth, outlinks_size, inlinks_size, nb_breadcrumbs, nb_aggregated_ratings, nb_ratings_values, nb_prices, nb_availabilities, nb_reviews, nb_reviews_count, nb_images, nb_search_in_url, nb_add_in_text, nb_filter_in_text, nb_search_in_text, nb_guide_achat_in_text, nb_product_info_in_text, nb_livraison_in_text, nb_garanties_in_text, nb_produits_similaires_in_text, nb_images_text, width_average, height_average, page_rank, page_type, concurrent_name, last_update, semantic_hits, semantic_title, inlinks_semantic, inlinks_semantic_count  from arbocrawl_results  where page_type !='Unknown' and concurrent_name = 'Cdiscount-maison' "; 
    catPred=["PAGE DEPTH AT SITE LEVEL","NUMBER OF OUTGOING LINKS","NUMBER OF INCOMING LINKS","NUMBER OF ITEMTYPE http://data-vocabulary.org/Breadcrumb","NUMBER OF ITEMPROP aggregateRating","NUMBER OF ITEMPROP ratingValue","NUMBER OF ITEMPROP price","NUMBER OF ITEMPROP availability","NUMBER OF ITEMPROP review","NUMBER OF ITEMPROP reviewCount","NUMBER OF ITEMPROP image","NUMBER OF OCCURENCES FOUND IN URL of search + recherche + Recherche + Search","NUMBER OF OCCURENCES FOUND IN PAGE TEXT ajout + ajouter + Ajout + Ajouter","NUMBER OF OCCURENCES FOUND IN PAGE TEXT filtre + facette + Filtre + Facette + filtré + filtrés","NUMBER OF OCCURENCES FOUND IN PAGE TEXT Ma recherche + Votre recherche + résultats pour + résultats associés","NUMBER OF OCCURENCES FOUND IN PAGE TEXT guide d""achat + Guide d""achat","NUMBER OF OCCURENCES FOUND IN PAGE TEXT caractéristique + Caractéristique + descriptif + Descriptif +information + Information","NUMBER OF OCCURENCES FOUND IN PAGE TEXT livraison + Livraison + frais de port + Frais de port","NUMBER OF OCCURENCES FOUND IN PAGE TEXT garantie + Garantie +assurance + Assurance","NUMBER OF OCCURENCES FOUND IN PAGE TEXT Produits Similaires + produits similaires + Meilleures Ventes + meilleures ventes +Meilleures ventes + Nouveautés + nouveautés + Nouveauté + nouveauté","NUMBER OF HTML TAG img IN THE PAGE","AVERAGE WIDTH OF HTML TAG img IN THE PAGE","AVERAGE HEIGHT OF HTML TAG img IN THE PAGE"];
    semPred =["PAGE TEXT", "PAGE TITLE", "PAGE H1", "PAGE SHORT DESCRIPTION","TEN BEST TF/IDF HITS FOR THE PAGE","TITLE TF/IDF","PAGE INCOMING LINKS ANCHOR SEMANTIC"];

    print "Executing the following request to fetch data for Cdiscount-maison from the ARBOCRAWL_RESULTS table : " + cdiscount_maison_request
    print"Page-type predictors : "+ ', '.join(catPred)
    print"Semantic predictors : " + ', '.join(semPred)

    df = pd.read_sql(cdiscount_maison_request, conn)
    
  
    url_list = df.url.values
    semantic_columns = ["url","title","h1","short_description","semantic_hits", "semantic_title", "inlinks_semantic"];
    semantic_predictors = df[list(semantic_columns)].values;
    
    classifying_columns = ["depth", "outlinks_size", "inlinks_size", "nb_breadcrumbs", "nb_aggregated_ratings", "nb_ratings_values", "nb_prices", "nb_availabilities", "nb_reviews", "nb_reviews_count", "nb_images", "nb_search_in_url", "nb_add_in_text", "nb_filter_in_text", "nb_search_in_text", "nb_guide_achat_in_text", "nb_product_info_in_text", "nb_livraison_in_text", "nb_garanties_in_text", "nb_produits_similaires_in_text", "nb_images_text", "width_average","height_average"]
    classifying_predictors = df[list(classifying_columns)].values;
    X= np.asanyarray(classifying_predictors);
    y = df.page_type.values;

    print type(X)
    print X.shape
    print type(y)
    print y.shape
    
    # fetching the data to predict
    to_predict_request = "select url, whole_text, title, h1, short_description, status_code, depth, outlinks_size, inlinks_size, nb_breadcrumbs, nb_aggregated_ratings, nb_ratings_values, nb_prices, nb_availabilities, nb_reviews, nb_reviews_count, nb_images, nb_search_in_url, nb_add_in_text, nb_filter_in_text, nb_search_in_text, nb_guide_achat_in_text, nb_product_info_in_text, nb_livraison_in_text, nb_garanties_in_text, nb_produits_similaires_in_text, nb_images_text, width_average, height_average, page_rank, page_type, concurrent_name, last_update, semantic_hits, semantic_title, inlinks_semantic, inlinks_semantic_count  from arbocrawl_results  where concurrent_name != 'Cdiscount-maison' "; 
    df_to_predict = pd.read_sql(to_predict_request, conn)
    # df_to_predict.dropna()
    # df_to_predict.replace([np.inf, -np.inf], np.nan).dropna(subset=list(classifying_columns), how="all")
    # df_to_predict.dropna(subset=list(classifying_columns), how="all", with_inf=True)
    # indexnan = sum(np.isnan(Xval))
    # indexinfinite = np.isfinite(Xval)
    classifying_predictors_to_predict = df_to_predict[list(classifying_columns)].values;
    Xval= np.asanyarray(classifying_predictors_to_predict);
    print type(Xval)
    print Xval.shape
    
    url_val_list = df_to_predict.url.values
    print type(url_val_list)
    print url_val_list.shape
    
    # we must here filter the NaN / Infinity in Xval values
    #print np.isnan(Xval)
    #Xval = Xval[~np.isnan(Xval)]
    #print Xval.shape
 
    # transforming the predictors / rescaling the predictors
    # we don't need to do that
    #X = StandardScaler().fit_transform(X)
    #Xval = StandardScaler().fit_transform(Xval)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    single_tree = DecisionTreeClassifier(max_depth=5)
    single_tree.fit(X_train, y_train)
    single_tree_score = single_tree.score(X_test, y_test)
    print "Single tree score " + str(single_tree_score)
    
    random_forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    random_forest.fit(X_train, y_train)
    random_forest_score = random_forest.score(X_test, y_test)
    print "Random forest score " + str(random_forest_score)
    
    kneighbors =  KNeighborsClassifier(3)
    kneighbors.fit(X_train, y_train)
    kneighbors_score = kneighbors.score(X_test, y_test)
    print "K-Neighbors score " + str(kneighbors_score)
    
    adaboost =  AdaBoostClassifier()
    adaboost.fit(X_train, y_train)
    adaboost_score = adaboost.score(X_test, y_test)
    print "Ada boost score " + str(adaboost_score)

    gaussian_nb =  GaussianNB()
    gaussian_nb.fit(X_train, y_train)
    gaussian_nb_score = gaussian_nb.score(X_test, y_test)
    print "gaussian mixtures score " + str(gaussian_nb_score)
    
    lda =  LDA()
    lda.fit(X_train, y_train)
    lda_nb_score = lda.score(X_test, y_test)
    print "linear discriminant score " + str(lda_nb_score)
    
    qda =  QDA()
    qda.fit(X_train, y_train)
    qda_nb_score = qda.score(X_test, y_test)
    print "quadratic discriminant score " + str(qda_nb_score)
    
    #SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),


    # we now predict the dataset from the other web sites with the best scoring trained classifier
    y_val_predicted = random_forest.predict(Xval);
    print type(y_val_predicted)
    print y_val_predicted.shape
    
    print type(url_val_list)
    print url_val_list.shape
    
    url_validation_list = url_val_list.tolist()
    y_val_predicted_list = y_val_predicted.tolist()

#    displaying the classified data    
#    pprint.pprint(y_val_predicted_list)
#    pprint.pprint(url_validation_list)
    classified_values = zip(url_validation_list, y_val_predicted_list)
    print "Updating the database with the classification results"
    update_database_with_page_type(conn, classified_values)
    conn.close()


def update_database_with_page_type(conn, classified_values):
    updating_request = "update arbocrawl_results set page_type=%s where url=%s";
    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor()
    for url, page_type in classified_values :
        print "Updating row url "+url+" with "+page_type
        cursor.execute(updating_request,(page_type,url)); 

    cursor.close()
    conn.commit()
       
def assign_enumerated_value(page_type): 
    if "FicheProduit" == page_type :
        return 0;
    if "Vitrine" == page_type :
        return 1;
    if "SearchDexing" == page_type :
        return 2;
    if "ListeProduit" == page_type :
        return 3;
    if "Unknown" == page_type :
        return 4;
    return 4;
 
if __name__ == "__main__":
    main()