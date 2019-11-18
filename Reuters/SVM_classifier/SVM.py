import time;
from gensim.models import Word2Vec
import pandas as pd
import time
from nltk.corpus import stopwords
import numpy as np
from KaggleWord2VecUtility import KaggleWord2VecUtility
from numpy import float32
import math
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn import svm
import pickle
import cPickle
from math import *
from sklearn.mixture import GaussianMixture
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame
import time;
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from KaggleWord2VecUtility import KaggleWord2VecUtility
from numpy import float32
import math
from sklearn.ensemble import RandomForestClassifier
import sys
from random import uniform
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import pickle
import cPickle
from math import *
from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn import grid_search
from sklearn.mixture import GMM
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import pickle, os
import pdb
from gensim.models.keyedvectors import KeyedVectors

WORD_EMBED_DIR = "../Word_Vectors"
INPUT_DIR = "../data/"
WORD_EMBED_DIR_DOC2VEC = "../Word_Vectors/doc2vecC"
WTV_DIR = "../Word_Topic_Vectors"
f = open('log.txt','a')

sparsity = 1
embedding_type = "doc2vec" #doc2vec and word2vec
multisense = 0

def create_cluster_vector_and_gwbowv(wtv, wordlist,  dimension, num_centroids, train=False):
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros(num_centroids * dimension, dtype="float32")
    global min_no
    global max_no

    for word in wordlist:
        try:
            temp = wtv[word]
        except:
            continue

        bag_of_centroids += wtv[word]

    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    if (norm != 0):
        bag_of_centroids /= norm

    # To make feature vector sparse, make note of minimum and maximum values.
    if train:
        min_no += min(bag_of_centroids)
        max_no += max(bag_of_centroids)

    return bag_of_centroids


filename = "reuters_doc2vecc_idx_proba_polysemy.txt"
num_clusters = 60
num_features = 200

# Create train and text data.
all = pd.read_pickle('../data/all.pkl')
lb = MultiLabelBinarizer()
Y = lb.fit_transform(all.tags)
train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)

train = DataFrame({'text': []})
test = DataFrame({'text': []})

train["text"] = train_data.reset_index(drop=True)
test["text"] = test_data.reset_index(drop=True)

for sparsity in [0,1]:
    for embedding_type in ["doc2vec","word2vec"]:
        start = time.time()
        wtv_name = "wtv_sparsity_"+str(sparsity)+"_word_embedding_type_"+str(embedding_type)+"_multisense_"+str(multisense)
        wtv = joblib.load(os.path.join(WTV_DIR,wtv_name))
        # gwbowv is a matrix which contain normalised normalised gwbowv.
        gwbowv = np.zeros((train["text"].size, num_clusters * (num_features)), dtype="float32")

        counter = 0

        min_no = 0
        max_no = 0
        start4 = time.time()
        for review in train["text"]:
            # Get the wordlist in each text article.
            words = KaggleWord2VecUtility.review_to_wordlist(review, \
                                                             remove_stopwords=True)
            #words = train_docs[counter]
            gwbowv[counter] = create_cluster_vector_and_gwbowv(wtv, words, num_features, num_clusters, train=True)
            counter += 1
            if counter % 1000 == 0:
                print "Train text Covered : ", counter
        start5 = time.time()
        f.write("time taken in building train document Vectors " + str(start5 - start4) + "\n")

        gwbowv_name = "SDV_sparsity_"+str(sparsity)+"_word_embedding_type_"+str(embedding_type)+"_multisense_"+str(multisense)+ "feature_matrix_gmm_sparse.npy"

        endtime_gwbowv = time.time() - start
        print "Created gwbowv_train: ", endtime_gwbowv, "seconds."

        gwbowv_test = np.zeros((test["text"].size, num_clusters * (num_features)), dtype="float32")

        counter = 0

        for review in test["text"]:
            # Get the wordlist in each text article.
            words = KaggleWord2VecUtility.review_to_wordlist(review, \
                                                             remove_stopwords=True)
            #words = test_docs[counter]

            gwbowv_test[counter] = create_cluster_vector_and_gwbowv(wtv, words, num_features, num_clusters)
            counter += 1
            if counter % 1000 == 0:
                print "Test Text Covered : ", counter

        test_gwbowv_name = "TEST_SDV_sparsity_"+str(sparsity)+"_word_embedding_type_"+str(embedding_type)+"_multisense_"+str(multisense)+"feature_matrix_gmm_sparse.npy"
        start6 = time.time()
        f.write("time taken in building test document Vectors " + str(start6 - start5) + "\n")

        print "Making sparse..."
        # Set the threshold percentage for making it sparse.
        percentage = 0.04
        min_no = min_no * 1.0 / len(train["text"])
        max_no = max_no * 1.0 / len(train["text"])
        print "Average min: ", min_no
        print "Average max: ", max_no
        thres = (abs(max_no) + abs(min_no)) / 2
        thres = thres * percentage

        # Make values of matrices which are less than threshold to zero.
        temp = abs(gwbowv) < thres
        #gwbowv[temp] = 0

        temp = abs(gwbowv_test) < thres
        #gwbowv_test[temp] = 0

        # saving gwbowv train and test matrices
        np.save(gwbowv_name, gwbowv)
        np.save(test_gwbowv_name, gwbowv_test)

        endtime = time.time() - start
        print "Total time taken: ", endtime, "seconds."

        print "********************************************************"


