'''
Created on 11 Jun 2015

@author: root
'''
#!/usr/bin/env python

#
# *************************************** #


# ****** Read the two training sets and the test set
#
import math
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import psycopg2
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    if nwords == 0 :
        print 'Trouble here where no words was found in the vocabulary'
    else :
        featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews


def getCleanDescriptions(descriptions):
    clean_descriptions = []
    local_counter=0
    for description in descriptions["description"]:
        clean_descriptions.append( KaggleWord2VecUtility.review_to_wordlist( description, remove_stopwords=True ))
        local_counter=local_counter+1
        print('Adding line : '+str(local_counter))
    return clean_descriptions


if __name__ == '__main__':

#    train = pd.read_csv( os.path.join(os.path.dircdiscount_uniformly_restrained_training.csvname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
#    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
#    unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
    conn_string = "host='localhost' dbname='CATEGORIZERDB' user='postgres' password='mogette'"
    # print the connection string we will use to connect
    print "Connecting to database\n    ->%s" % (conn_string)
 
    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)
    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor()
    # execute our Query
    # X = np.asarray(predictors_list);
    my_idmapping_request = "select * from CATEGORY_MAPPING"

   # fetching data to display for magasin Musique
    cursor.execute(my_idmapping_request); 
    mapping_data = cursor.fetchall()
    my_mapping_dictionnary = {}
    for element in mapping_data:
        my_mapping_dictionnary[element[1]]=element[0]

    training_data_path = '/home/sduprey/My_Data/My_Cdiscount_Challenge/light_cdiscount_uniformly_restrained_training.csv'
    testing_data_path  = '/home/sduprey/My_Data/My_Cdiscount_Challenge/light_cdiscount_uniformly_restrained_testing.csv'

    # Read data from files
   
    train = pd.read_csv(training_data_path, header=0, delimiter=";", quoting=3 )
    test = pd.read_csv(testing_data_path, header=0, delimiter=";", quoting=3 )

    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
     " end of file \n" % (train["description"].size,
     test["description"].size )



    # Load the punkt tokenizer
#    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')




    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    counter=0
    print "Parsing sentences from training set"
    for review in train["description"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        counter=counter+1
        print('Adding sentences : '+str(counter))
        
    for test_review in test["description"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(test_review, tokenizer)
        counter=counter+1
        print('Adding sentences : '+str(counter))

#    print "Parsing sentences from unlabeled set"
#    for review in unlabeled_train["review"]:
#        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
#        counter=counter+1
#        print('Adding sentences : '+str(counter))

    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "light_300features_40minwords_10context"
    model.save(model_name)

   #  model.doesnt_match("homme femme enfant cuisine".split())
   #  model.doesnt_match("france angleterre allemagne".split())
   #  model.doesnt_match("paris berlin londres".split())
   #  model.most_similar("homme")
   # model.most_similar("reine")
   # model.most_similar("terrible")

    # ****** Create average vectors for the training and test sets
    #
    print "Creating average feature vecs for training reviews"

    trainDataVecs = getAvgFeatureVecs( getCleanDescriptions(train), model, num_features )

    print "Creating average feature vecs for test reviews"

    testDataVecs = getAvgFeatureVecs( getCleanDescriptions(test), model, num_features )

    
#    print type(trainDataVecs)
#    print trainDataVecs.shape
#    print trainDataVecs.dtype
#    print trainDataVecs.size
#    for row in trainDataVecs:
#        print row    
#    for element in trainDataVecs.flat:
#        print element
#        if (math.isnan(element) or math.isinf(element)):
#            print 'Big trouble'
        
    # x[:, ~np.isnan(x).any(axis=0)]
    
    # ****** Fit a random forest to the training set, then make predictions
    #
    # Fit a random forest to the training data, using 100 trees
    
    
    
        # we here list the different classifiers we want to test
    names = ["Decision Tree","AdaBoost", "Nearest Neighbors", "Naive Bayes","Random Forest", "RBF SVM", "LDA","Linear SVM", "QDA"]
    classifiers = [
    DecisionTreeClassifier(max_depth=5),
    AdaBoostClassifier(),
    KNeighborsClassifier(3),
    GaussianNB(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    SVC(gamma=2, C=1),
    LDA(),
    SVC(kernel="linear", C=0.025),
    QDA()]
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print "Training " + name +" to calibrate our classifier"
        clf.fit(trainDataVecs, train["output"])
        #score = clf.score(X_test, y_test)
        print name+" has been trained, we now make the prediction"

        Id_Categorie=[]
        # Test & extract results
        result = clf.predict(testDataVecs)
        for element in result.flat:
            Id_Categorie.append(my_mapping_dictionnary[element])
#        print element
        # Write the test results
        
        for id, cat in zip(test["id"],Id_Categorie):
            print 'Id_Produit:'+str(id) +';Id_Categorie'+str(Id_Categorie) 
        
        output = pd.DataFrame( data={"Id_Produit":test["id"], "Id_Categorie":Id_Categorie} )
        output.to_csv('/home/sduprey/My_Data/My_Outgoing_Data/My_Cdiscount_Challenge/light_'+ name +'Word2Vec_AverageVectors.csv', index=False, quoting=3)
        print 'Wrote /home/sduprey/My_Data/My_Outgoing_Data/My_Cdiscount_Challenge/light_'+ name +'Word2Vec_AverageVectors.csv'
        
        
        
        
