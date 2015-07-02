#!/usr/bin/python
# coding: utf-8
'''
Created on 26 Jan 2015

@author: sduprey
'''
import psycopg2
import sys
import pprint
import numpy as np
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier

def main():
    #Define our connection string
    conn_string = "host='localhost' dbname='CRAWL4J' user='postgres' password='mogette'"
    # print the connection string we will use to connect
    print "Connecting to database\n    ->%s" % (conn_string)
 
    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)
 
    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor()
 
    # execute our Query
    # X = np.asarray(predictors_list);
    
    my_request = "select url, whole_text, title, h1, short_description, status_code, depth, outlinks_size, inlinks_size, nb_breadcrumbs, nb_aggregated_ratings, nb_ratings_values, nb_prices, nb_availabilities, nb_reviews, nb_reviews_count, nb_images, nb_search_in_url, nb_add_in_text, nb_filter_in_text, nb_search_in_text, nb_guide_achat_in_text, nb_product_info_in_text, nb_livraison_in_text, nb_garanties_in_text, nb_produits_similaires_in_text, nb_images_text, width_average, height_average, page_rank, page_type, concurrent_name, last_update, semantic_hits, semantic_title, inlinks_semantic, inlinks_semantic_count  from arbocrawl_results  where concurrent_name = (%s) "; 
    #url 0, whole_text 1, title 2, h1 3, short_description 4, status_code 5, depth 6, outlinks_size 7, inlinks_size 8, nb_breadcrumbs 9, nb_aggregated_ratings 10, nb_ratings_values 11, nb_prices 12, nb_availabilities 13, nb_reviews 14, nb_reviews_count 15, nb_images 16, nb_search_in_url 17, nb_add_in_text 18, nb_filter_in_text 19, nb_search_in_text 20, nb_guide_achat_in_text 21, nb_product_info_in_text 22, nb_livraison_in_text 23, nb_garanties_in_text 24, nb_produits_similaires_in_text 25, nb_images_text 26, width_average 27, height_average 28, page_rank 29, page_type 30, concurrent_name 31, last_update 32, semantic_hits 33, semantic_title 34, inlinks_semantic 35, inlinks_semantic_count 36  from arbocrawl_results 
    catPred=["PAGE DEPTH AT SITE LEVEL","NUMBER OF OUTGOING LINKS","NUMBER OF INCOMING LINKS","NUMBER OF ITEMTYPE http://data-vocabulary.org/Breadcrumb","NUMBER OF ITEMPROP aggregateRating","NUMBER OF ITEMPROP ratingValue","NUMBER OF ITEMPROP price","NUMBER OF ITEMPROP availability","NUMBER OF ITEMPROP review","NUMBER OF ITEMPROP reviewCount","NUMBER OF ITEMPROP image","NUMBER OF OCCURENCES FOUND IN URL of search + recherche + Recherche + Search","NUMBER OF OCCURENCES FOUND IN PAGE TEXT ajout + ajouter + Ajout + Ajouter","NUMBER OF OCCURENCES FOUND IN PAGE TEXT filtre + facette + Filtre + Facette + filtré + filtrés","NUMBER OF OCCURENCES FOUND IN PAGE TEXT Ma recherche + Votre recherche + résultats pour + résultats associés","NUMBER OF OCCURENCES FOUND IN PAGE TEXT guide d""achat + Guide d""achat","NUMBER OF OCCURENCES FOUND IN PAGE TEXT caractéristique + Caractéristique + descriptif + Descriptif +information + Information","NUMBER OF OCCURENCES FOUND IN PAGE TEXT livraison + Livraison + frais de port + Frais de port","NUMBER OF OCCURENCES FOUND IN PAGE TEXT garantie + Garantie +assurance + Assurance","NUMBER OF OCCURENCES FOUND IN PAGE TEXT Produits Similaires + produits similaires + Meilleures Ventes + meilleures ventes +Meilleures ventes + Nouveautés + nouveautés + Nouveauté + nouveauté","NUMBER OF HTML TAG img IN THE PAGE","AVERAGE WIDTH OF HTML TAG img IN THE PAGE","AVERAGE HEIGHT OF HTML TAG img IN THE PAGE"];
    semPred =["PAGE TEXT", "PAGE TITLE", "PAGE H1", "PAGE SHORT DESCRIPTION","TEN BEST TF/IDF HITS FOR THE PAGE","TITLE TF/IDF","PAGE INCOMING LINKS ANCHOR SEMANTIC"];

    print "Executing the following request to fetch data for Cdiscount-maison from the ARBOCRAWL_RESULTS table : " + my_request
    print"Page-type predictors : "+ ', '.join(catPred)
    print"Semantic predictors : " + ', '.join(semPred)
    

    # fetching training data from Cdiscount-maison
    my_filtered_request = "select url, whole_text, title, h1, short_description, status_code, depth, outlinks_size, inlinks_size, nb_breadcrumbs, nb_aggregated_ratings, nb_ratings_values, nb_prices, nb_availabilities, nb_reviews, nb_reviews_count, nb_images, nb_search_in_url, nb_add_in_text, nb_filter_in_text, nb_search_in_text, nb_guide_achat_in_text, nb_product_info_in_text, nb_livraison_in_text, nb_garanties_in_text, nb_produits_similaires_in_text, nb_images_text, width_average, height_average, page_rank, page_type, concurrent_name, last_update, semantic_hits, semantic_title, inlinks_semantic, inlinks_semantic_count  from arbocrawl_results  where page_type !='Unknown' and concurrent_name = (%s) "; 
    cursor.execute(my_filtered_request,("Cdiscount-maison",)); 
    # retrieve the records from the database
    records = cursor.fetchall()
    url_list = [item[0] for item in records];
    semantic_list =  [(item[1],item[2],item[3],item[4],item[33],item[34],item[35]) for item in records];
    predictor_list = [(item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],item[22],item[23],item[24],item[25],item[26],item[27],item[28]) for item in records];
    output_list    = [item[30] for item in records];
    y=[assign_enumerated_value(output) for output in output_list]
    X= np.asanyarray(predictor_list);
    y= np.asanyarray(y);
    print type(X)
    print X.shape
    print type(y)
    print y.shape
    
    # fetching the data to predict
    my_to_predict_request = "select url, whole_text, title, h1, short_description, status_code, depth, outlinks_size, inlinks_size, nb_breadcrumbs, nb_aggregated_ratings, nb_ratings_values, nb_prices, nb_availabilities, nb_reviews, nb_reviews_count, nb_images, nb_search_in_url, nb_add_in_text, nb_filter_in_text, nb_search_in_text, nb_guide_achat_in_text, nb_product_info_in_text, nb_livraison_in_text, nb_garanties_in_text, nb_produits_similaires_in_text, nb_images_text, width_average, height_average, page_rank, page_type, concurrent_name, last_update, semantic_hits, semantic_title, inlinks_semantic, inlinks_semantic_count  from arbocrawl_results  where concurrent_name != (%s) "; 
    cursor.execute(my_to_predict_request,("Cdiscount-maison",)); 
    # retrieve the records from the database
    records_to_validate = cursor.fetchall()
    url_to_validate_list = [item[0] for item in records_to_validate];
    semantic_to_validate_list =  [(item[1],item[2],item[3],item[4],item[33],item[34],item[35]) for item in records_to_validate];
    predictor_to_validate_list = [(item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21],item[22],item[23],item[24],item[25],item[26],item[27],item[28]) for item in records_to_validate];
    output_to_validate_list    = [item[30] for item in records_to_validate];
    
    Xval= np.asanyarray(predictor_to_validate_list);
    print type(Xval)
    print Xval.shape
    # we must here filter the NaN / Infinity in Xval values
    print np.isnan(Xval)
    Xval = Xval[~np.isnan(Xval)]
    print Xval.shape
 
    # transforming the predictors / rescaling the predictors
    # we don't need to do that
    #X = StandardScaler().fit_transform(X)
    #Xval = StandardScaler().fit_transform(Xval)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    single_tree = DecisionTreeClassifier(max_depth=5)
    single_tree.fit(X, output_list)
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
    pprint.pprint(y_val_predicted);
    
    # print out the records using pretty print
    # note that the NAMES of the columns are not shown, instead just indexes.
    # for most people this isn't very useful so we'll show you how to return
    # columns as a dictionary (hash) in the next example.
    #pprint.pprint(url_list)
    #pprint.pprint(semantic_list)
    #pprint.pprint(predictor_list)
    #pprint.pprint(output_list)

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