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


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def cluster_GMM(num_clusters, word_vectors):
    # Initalize a GMM object and use it for clustering.
    clf = GaussianMixture(n_components=num_clusters,
                          covariance_type="tied", init_params='kmeans', max_iter=50)
    # Get cluster assignments.
    clf.fit(word_vectors)
    idx = clf.predict(word_vectors)
    print "Clustering Done...", time.time() - start, "seconds"
    # Get probabilities of cluster assignments.
    idx_proba = clf.predict_proba(word_vectors)
    # Dump cluster assignments and probability of cluster assignments.
    joblib.dump(idx, 'gmm_latestclusmodel_len2alldata.pkl')
    print "Cluster Assignments Saved..."

    joblib.dump(idx_proba, 'gmm_prob_latestclusmodel_len2alldata.pkl')
    print "Probabilities of Cluster Assignments Saved..."
    return (idx, idx_proba)


def read_GMM(idx_name, idx_proba_name):
    # Loads cluster assignments and probability of cluster assignments.
    idx = joblib.load(idx_name)
    idx_proba = joblib.load(idx_proba_name)
    print "Cluster Model Loaded..."
    return (idx, idx_proba)


def get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict):
    # This function computes probability word-cluster vectors.

    prob_wordvecs = {}
    for word in word_centroid_map:
        prob_wordvecs[word] = np.zeros(num_clusters * num_features, dtype="float32")
        for index in range(0, num_clusters):
            try:
                prob_wordvecs[word][index * num_features:(index + 1) * num_features] = model[word] * \
                                                                                       word_centroid_prob_map[word][
                                                                                           index] * \
                                                                                       word_idf_dict[word]
            except:
                pass
    # prob_wordvecs_idf_len2alldata = {}

    # i = 0
    # for word in featurenames:
    # 	i += 1
    # 	if word in word_centroid_map:
    # 		prob_wordvecs_idf_len2alldata[word] = {}
    # 		for index in range(0, num_clusters):
    # 				prob_wordvecs_idf_len2alldata[word][index] = model[word] * word_centroid_prob_map[word][index] * word_idf_dict[word]

    # for word in prob_wordvecs_idf_len2alldata.keys():
    # 	prob_wordvecs[word] = prob_wordvecs_idf_len2alldata[word][0]
    # 	for index in prob_wordvecs_idf_len2alldata[word].keys():
    # 		if index==0:
    # 			continue
    # 		prob_wordvecs[word] = np.concatenate((prob_wordvecs[word], prob_wordvecs_idf_len2alldata[word][index]), axis=1)

    return prob_wordvecs


def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist, word_centroid_map, word_centroid_prob_map, dimension,
                                     word_idf_dict, featurenames, num_centroids, train=False):
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros(num_centroids * dimension, dtype="float32")
    global min_no
    global max_no

    for word in wordlist:
        try:
            temp = word_centroid_map[word]
        except:
            continue

        bag_of_centroids += prob_wordvecs[word]

    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    if (norm != 0):
        bag_of_centroids /= norm

    # To make feature vector sparse, make note of minimum and maximum values.
    if train:
        min_no += min(bag_of_centroids)
        max_no += max(bag_of_centroids)

    return bag_of_centroids


filename = "reuters_idxproba.txt"
for _ in range(5):
    start = time.time()

    num_features = 200  # Word vector dimensionality
    min_word_count = 20  # Minimum word count
    num_workers = 40  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    model_name = str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(
        context) + "context_len2alldata"
    # Load the trained Word2Vec model.
    model = Word2Vec.load(model_name)
    # Get wordvectors for all words in vocabulary.
    word_vectors = model.wv.syn0

    all = pd.read_pickle('all.pkl')
    start1 = time.time()
    start = time.time()
    # Set number of clusters.
    num_clusters = 60
    idx, idx_proba = cluster_GMM(num_clusters, word_vectors)
    idx_proba[idx_proba<0.2]=0
    n_clusteri = num_clusters
    f = open(filename, 'a')
    print "number of k clusters ", str(n_clusteri)
    f.write("number of k clusters " + str(n_clusteri) + "\n")
    start2 = time.time()
    f.write("time taken in clustering " + str(start2 - start1) + "\n")
    # Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
    # idx_name = "gmm_latestclusmodel_len2alldata.pkl"
    # idx_proba_name = "gmm_prob_latestclusmodel_len2alldata.pkl"
    # idx, idx_proba = read_GMM(idx_name, idx_proba_name)

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    # Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
    # list of probabilities of cluster assignments.
    word_centroid_prob_map = dict(zip(model.wv.index2word, idx_proba))

    # Computing tf-idf values.
    traindata = []
    for i in range(0, len(all["text"])):
        traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(all["text"][i], True)))

    tfv = TfidfVectorizer(strip_accents='unicode', dtype=np.float32)
    tfidfmatrix_traindata = tfv.fit_transform(traindata)
    featurenames = tfv.get_feature_names()
    idf = tfv._tfidf.idf_

    # Creating a dictionary with word mapped to its idf value
    print "Creating word-idf dictionary for Training set..."

    word_idf_dict = {}
    for pair in zip(featurenames, idf):
        word_idf_dict[pair[0]] = pair[1]
    start3 = time.time()
    f.write("time taken in idf processing " + str(start3 - start2) + "\n")
    # Pre-computing probability word-cluster vectors.
    prob_wordvecs = get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict)
    start4 = time.time()
    f.write("time taken in Creating word topic Vectors " + str(start4 - start3) + "\n")

    temp_time = time.time() - start
    print "Creating Document Vectors...:", temp_time, "seconds."

    # Create train and text data.
    lb = MultiLabelBinarizer()
    Y = lb.fit_transform(all.tags)
    train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)

    train = DataFrame({'text': []})
    test = DataFrame({'text': []})

    train["text"] = train_data.reset_index(drop=True)
    test["text"] = test_data.reset_index(drop=True)

    # gwbowv is a matrix which contain normalised normalised gwbowv.
    gwbowv = np.zeros((train["text"].size, num_clusters * (num_features)), dtype="float32")

    counter = 0

    min_no = 0
    max_no = 0
    for review in train["text"]:
        # Get the wordlist in each text article.
        words = KaggleWord2VecUtility.review_to_wordlist(review, \
                                                         remove_stopwords=True)
        gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map,
                                                           word_centroid_prob_map, num_features, word_idf_dict,
                                                           featurenames, num_clusters, train=True)
        counter += 1
        if counter % 1000 == 0:
            print "Train text Covered : ", counter
    start5 = time.time()
    f.write("time taken in building train document Vectors " + str(start5 - start4) + "\n")

    gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"

    endtime_gwbowv = time.time() - start
    print "Created gwbowv_train: ", endtime_gwbowv, "seconds."

    gwbowv_test = np.zeros((test["text"].size, num_clusters * (num_features)), dtype="float32")

    counter = 0

    for review in test["text"]:
        # Get the wordlist in each text article.
        words = KaggleWord2VecUtility.review_to_wordlist(review, \
                                                         remove_stopwords=True)
        gwbowv_test[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map,
                                                                word_centroid_prob_map, num_features, word_idf_dict,
                                                                featurenames, num_clusters)
        counter += 1
        if counter % 1000 == 0:
            print "Test Text Covered : ", counter

    test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(
        num_features) + "feature_matrix_gmm_sparse.npy"
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
    # np.save(gwbowv_name, gwbowv)
    # np.save(test_gwbowv_name, gwbowv_test)

    endtime = time.time() - start
    print "Total time taken: ", endtime, "seconds."

    print "********************************************************"


    def drange(start, stop, step):
        r = start
        while r < stop:
            yield r
            r += step


    print "Fitting One vs Rest SVM classifier to labeled cluster training data..."

    start = time.time()

    num_features = 200
    num_clusters = 60

    all = pd.read_pickle('all.pkl')

    # Get train and text data.
    lb = MultiLabelBinarizer()
    Y = lb.fit_transform(all.tags)
    train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)

    train = DataFrame({'text': []})
    test = DataFrame({'text': []})

    train["text"] = train_data.reset_index(drop=True)
    test["text"] = test_data.reset_index(drop=True)

    # Load feature vectors.
    gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"
    # gwbowv = np.load(gwbowv_name)

    test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(
        num_features) + "feature_matrix_gmm_sparse.npy"
    # gwbowv_test = np.load(test_gwbowv_name)

    param_grid = [
        {'estimator__C': np.arange(0.1, 150, 50)}
        # {'C': [1, 10, 100, 200, 500, 1000], 'gamma': [0.01, 0.05, 0.001, 0.005,  0.0001], 'kernel': ['rbf']},
    ]
    import gc

    del prob_wordvecs, idx_proba, all, train["text"], word_vectors, model, test["text"], train_data, test_data
    gc.collect()
    scores = ['f1_weighted']  # , 'accuracy', 'recall', 'f1']
    for score in scores:
        strt = time.time()
        print "# Tuning hyper-parameters for", score, "\n"
        clf = GridSearchCV(OneVsRestClassifier(LogisticRegression(C=100.0), n_jobs=1), param_grid, cv=5, n_jobs=10,
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
            f.write("Precision@ " + str(k) + " : " + str(Total_Precision * 1.0 / loop_var) + "\n")
            f.write("NDCG@ " + str(k) + " : " + str(Total_DCG * 1.0 / loop_var) + "\n")
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
        f.write("Weighted F1score: " + str(f1_score(Y_test, pred, average='weighted')) + "\n")
        # print "Total time taken: ", time.time()-start, "seconds."
    import gc

    del gwbowv, gwbowv_test, clf
    gc.collect()
    endtime = time.time()
    print "Total time taken: ", endtime - start, "seconds."

    print "********************************************************"
