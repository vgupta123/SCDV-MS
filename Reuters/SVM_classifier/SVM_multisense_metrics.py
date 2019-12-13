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
import time;
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from KaggleWord2VecUtility import KaggleWord2VecUtility
from numpy import float32
import math,os
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


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


print "Fitting One vs Rest SVM classifier to labeled cluster training data..."

start = time.time()
num_features = 200#int(sys.argv[1])
num_clusters = 60#int(sys.argv[2])

WORD_EMBED_DIR = "../Word_Vectors"
INPUT_DIR = "../data/"
WORD_EMBED_DIR_DOC2VEC = "../Word_Vectors/doc2vecC"
WTV_DIR = "../Word_Topic_Vectors"
f = open('log.txt','a')

sparsity = 1
embedding_type = "doc2vec" #doc2vec and word2vec
multisense = 1

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
train_docs, Y_train,test_docs,Y_test = pickle.load(open(os.path.join(INPUT_DIR, "train_text_annotate_sets.pkl"),"rb"))
train["text"] = train_data.reset_index(drop=True)
test["text"] = test_data.reset_index(drop=True)

for sparsity in [1]:
    for embedding_type in ["doc2vec"]:#,"word2vec"]:
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
            #words1 = KaggleWord2VecUtility.review_to_wordlist(review, \
            #                                                 remove_stopwords=True)
            words = train_docs[counter]
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
            #words = KaggleWord2VecUtility.review_to_wordlist(review, \
            #                                                 remove_stopwords=True)
            words = test_docs[counter]

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

        param_grid = [
            {'estimator__C': np.arange(0.1, 150, 10)}
            # {'C': [1, 10, 100, 200, 500, 1000], 'gamma': [0.01, 0.05, 0.001, 0.005,  0.0001], 'kernel': ['rbf']},
        ]
        scores = ['f1_weighted']  # , 'accuracy', 'recall', 'f1']
        f = open("results.txt", "a")
        f.write(str(gwbowv_name) + "\n")
        for score in scores:
            strt = time.time()
            print "# Tuning hyper-parameters for", score, "\n"
            clf = GridSearchCV(OneVsRestClassifier(LogisticRegression(C=100.0), n_jobs=30), param_grid, cv=5, n_jobs=30,
                               scoring='%s' % score)
            clf = clf.fit(gwbowv, Y_train)

            pred = clf.predict(gwbowv_test)
            pred_proba = clf.predict_proba(gwbowv_test)

            K = [1, 3, 5]

            for k in K:
                Total_Precision = 0
                Total_DCG = 0
                norm = 0
                for i in range(k):
                    norm += 1 / math.log(i + 2)

                loop_var = 0
                for item in pred_proba:
                    classelements = sorted(range(len(item)), key=lambda i: item[i])[-k:]
                    classelements.reverse()
                    precision = 0
                    dcg = 0
                    loop_var2 = 0
                    for element in classelements:
                        if Y_test[loop_var][element] == 1:
                            precision += 1
                            dcg += 1 / math.log(loop_var2 + 2)
                        loop_var2 += 1

                    Total_Precision += precision * 1.0 / k
                    Total_DCG += dcg * 1.0 / norm
                    loop_var += 1
                print "Precision@", k, ": ", Total_Precision * 1.0 / loop_var
                print "NDCG@", k, ": ", Total_DCG * 1.0 / loop_var
                f.write("Precision@" + str(k) + ": " + str(Total_Precision * 1.0 / loop_var) + "\n")
                f.write("NDCG@" + str(k) + ": " + str(Total_DCG * 1.0 / loop_var) + "\n")
            print "Coverage Error: ", coverage_error(Y_test, pred_proba)
            print "Label Ranking Average precision score: ", label_ranking_average_precision_score(Y_test, pred_proba)
            print "Label Ranking Loss: ", label_ranking_loss(Y_test, pred_proba)
            print "Hamming Loss: ", hamming_loss(Y_test, pred)
            print "Weighted F1score: ", f1_score(Y_test, pred, average='weighted')
            f.write("Coverage Error: " + str(coverage_error(Y_test, pred_proba)) + "\n")
            f.write("Label Ranking Average precision score: " + str(
                label_ranking_average_precision_score(Y_test, pred_proba)) + "\n")
            f.write("Label Ranking Loss: " + str(label_ranking_loss(Y_test, pred_proba)) + "\n")
            f.write("Hamming Loss: " + str(hamming_loss(Y_test, pred)) + "\n")
            f.write("Weighted F1score: " + str(f1_score(Y_test, pred, average='weighted')) + "\n\n\n")
            # print "Total time taken: ", time.time()-start, "seconds."

        endtime = time.time()
        print "Total time taken: ", endtime - start, "seconds."

        print "********************************************************"




