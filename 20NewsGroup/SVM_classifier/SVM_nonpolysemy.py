import time
import warnings
from gensim.models import Word2Vec
import pandas as pd
import pdb
import time
from nltk.corpus import stopwords
import numpy as np
from KaggleWord2VecUtility import KaggleWord2VecUtility
from gensim.models.keyedvectors import KeyedVectors
from numpy import float32
import math
import pickle
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.svm import SVC, LinearSVC
import pickle
from math import *
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from scipy.sparse import csr_matrix as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as normsp
import os   

WORD_EMBED_DIR = "../Word_Vectors"
INPUT_DIR = "../data/"
WTV_DIR = "../Word_Topic_Vectors"
f = open('log.txt','a')

def create_cluster_vector_and_gwbowv(wtv,wordlist,final_dimension,train=False):
    
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros( final_dimension , dtype="float32" )
    
    global min_no
    global max_no

    for word in wordlist:
        try:
            temp = wtv[word]
        except:
            continue
        bag_of_centroids += wtv[word]

    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))

    if(norm!=0):
        bag_of_centroids /= norm

    # To make feature vector sparse, make note of minimum and maximum values.
    if train:
        min_no += min(bag_of_centroids)
        max_no += max(bag_of_centroids)

    return bag_of_centroids

c = 0
def create_cluster_vector_and_gwbowv_sparse(wtv,wordlist,final_dimension,train=False):
    global c
    # This function computes SDV feature vectors.
    bag_of_centroids = sp(np.zeros( final_dimension , dtype="float32" ))
    # print(bag_of_centroids.shape)
    global min_no
    global max_no

    for word in wordlist:
        try:
            temp = wtv[word]
        except:
            continue
        # print(word)
        # print("word = ",wtv[word].shape)
        # print(bag_of_centroids.shape,wtv[word].shape)
        bag_of_centroids = bag_of_centroids + wtv[word]
        # bag_of_centroids.sum(wtv[word])
    # print(len(bag_of_centroids.nonzero()[0]))
    c+= len(bag_of_centroids.nonzero()[0])
    # norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    norm = normsp(bag_of_centroids)
    bag_of_centroids.multiply(1/norm)
    # if(norm!=0):
        # bag_of_centroids /= norm

    # To make feature vector sparse, make note of minimum and maximum values.
    # if train:
    #     min_no += min(bag_of_centroids)
    #     max_no += max(bag_of_centroids)
    # bag_of_centroids = bag_of_centroids.toarray()
    return bag_of_centroids



if len(sys.argv) == 3 :
    print("Number of clusters = ",sys.argv[2])
    print("Word embedding dimension = ",sys.argv[1])
    final_dimension = int(sys.argv[1])*int(sys.argv[2])
    print("Dimension of WTV = ",final_dimension)
    reduction = 0
else :
    final_dimension = int(sys.argv[1])
    reduction = 1
    print("Dimension of WTV = ",final_dimension)



if reduction == 0 :
    name = "wtv_nonpolysemy"
    print("Using the non-reduced WTV file ( wtv_nonpolysemy)")
else:
    name = "reduced_wtv_nonpolysemy"
    print("Using the reduced WTV file (reduced_wtv_nonpolysemy)")



print("Creating Document Vectors...:")
# gwbowv is a matrix which contains normalised document vectors.

wtv = joblib.load(os.path.join(WTV_DIR,name))

# c = 0
# for i in wtv :
#     c+=np.count_nonzero(wtv[i]) 
#     wtv[i] = sp(wtv[i])
    # print()
# print("percent non zero = ",(c/(final_dimension*len(wtv))*100))

# num_features = 200
# num_clusters = 60
# final_dimension = 2000

for i in wtv:
    # print(wtv[i].shape)
    assert ( final_dimension == wtv[i].shape[0])

counter = 0
min_no = 0
max_no = 0


start = time.time()

train = pd.read_csv(os.path.join(INPUT_DIR,'train_v2.tsv'), header=0, delimiter="\t")
# Load test data.
test = pd.read_csv(os.path.join(INPUT_DIR,'test_v2.tsv'), header=0, delimiter="\t")
all = pd.read_csv(os.path.join(INPUT_DIR,'all_v2.tsv'), header=0, delimiter="\t")

e = open(os.path.join(INPUT_DIR,'20_newsgroup_tokenized_multisense_sentences_1000_train.txt'),"r")
g = open(os.path.join(INPUT_DIR,"20_newsgroup_tokenized_multisense_sentences_1000_test.txt"),"r")
train_lines = e.readlines()
train_lines_mod = [line[:-1].split() for line in train_lines]
test_lines = g.readlines()
test_lines_mod = [line[:-1].split() for line in test_lines]

ft = time.time()
gwbowv = np.zeros( (train["news"].size, final_dimension ), dtype="float32")
# gwbowv = {}
cnt = 0
for review in train["news"]:
    # Get the wordlist in each news article.
    # words = train_lines_mod[counter]
    words = review
    s = time.time()
    t = create_cluster_vector_and_gwbowv(wtv,words,final_dimension,train=True)
    cnt += time.time()-s
    gwbowv[counter] = t
    counter+=1
    if counter % 1000 == 0:
        print("Train News Covered : ",counter)

gwbowv_name = "SDV_" + str(final_dimension) + "feature_matrix_gmm_sparse.npy"
start2 = time.time()
f.write("time taken in building train document Vectors " +str(start2-start)+"\n")

gwbowv_test = np.zeros( (test["news"].size, final_dimension ), dtype="float32")
# gwbowv_test = {}
counter = 0

for review in test["news"]:
    # Get the wordlist in each news article.
    # words = test_lines_mod[counter]
    words = review
    s = time.time()
    t = create_cluster_vector_and_gwbowv(wtv,words,final_dimension)
    cnt += time.time()-s
    gwbowv_test[counter] = t
    counter+=1
    if counter % 1000 == 0:
        print("Test News Covered : ",counter)


test_gwbowv_name = "TEST_SDV_" + str(final_dimension) + "feature_matrix_gmm_sparse.npy"
f.write("time taken in building test document Vectors " +str(time.time()-start2)+"\n")
print("time taken in building test document Vectors " +str(time.time()-start2)+"\n")
# start =time.time()
print("time for creation of document vecotrs both train and test = ",time.time()-ft)
print("creation time : ",cnt)
# for i in wtv:
#   wtv[i] = wtv[i].toarray()

# print("Making sparse...")
# Set the threshold percentage for making it sparse. 
# percentage = 0.04
# min_no = min_no*1.0/len(train["news"])
# max_no = max_no*1.0/len(train["news"])
# # print("Average min: ", min_no)
# # print("Average max: ", max_no)
# thres = (abs(max_no) + abs(min_no))/2
# thres = thres*percentage

# Make values of matrices which are less than threshold to zero.
#temp = abs(gwbowv) < thres
#gwbowv[temp] = 0

#temp = abs(gwbowv_test) < thres
#gwbowv_test[temp] = 0

#saving gwbowv train and test matrices
#np.save(gwbowv_name, gwbowv)
# #np.save(test_gwbowv_name, gwbowv_test)
print("dimension of document vectors : ",gwbowv.shape)
print("percent of non zero entries in document vectors", np.count_nonzero(gwbowv)/(gwbowv.shape[0]*gwbowv.shape[1]))
print("dimension of test document vectors : ",gwbowv_test.shape)
print("percent of non zero entries in test document vectors", np.count_nonzero(gwbowv_test)/(gwbowv_test.shape[0]*gwbowv_test.shape[1]))
print(c/(test["news"].size * final_dimension))
endtime = time.time() - start
# print("SDV created and dumped: ", endtime, "seconds.")
print("Fitting a SVM classifier on labeled training data...")

param_grid = [
      {'C': np.arange(1.1, 5, 0.1)}]
      # {'C': np.arange(5, 10, 0.3)}]

scores = ['f1_weighted']#, 'recall_micro', 'f1_micro' , 'precision_micro', 'recall_macro', 'f1_macro' , 'precision_macro', 'recall_weighted', 'f1_weighted' , 'precision_weighted'] #, 'accuracy', 'recall', 'f1']
for score in scores:
        strt = time.time()
        print("# Tuning hyper-parameters for", score, "\n")
        clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=5,n_jobs=2,scoring= '%s' % score)
        clf.fit(gwbowv, train["class"])
        print("Best parameters set found on development set:\n")
        C_score = clf.best_params_
        print(C_score)
        f.write("Best parameters set found on development set: "+str(C_score)+"\n")
        joblib.dump(clf,'clf')
        print("Best value for ", score, ":\n")
        start7 =time.time()
        f.write("time taken in classifying documents " +str(start7-strt)+"\n")
        f.write("time taken total " +str(start7-start)+"\n")
        # f.write("number of k clusters " +str(num_clusters)+"\n")
        print(clf.best_score_)
        t1 = time.time()
        Y_true, Y_pred  = test["class"], clf.predict(gwbowv_test)
        print("predicton time ",time.time()-t1)
        print("Report")
        classification_reporting = classification_report(Y_true, Y_pred, digits=6)
        print(classification_reporting)
        f.write(classification_reporting)
        print("Accuracy: ",clf.score(gwbowv_test,test["class"]))
        f.write("Accuracy: "+str(clf.score(gwbowv_test,test["class"]))+"\n")
        print("Time taken:", time.time() - strt, "\n")
endtime = time.time()
print("Total time taken: ", endtime-start, "seconds." )
import gc
del gwbowv, gwbowv_test,wtv, clf
gc.collect()
print("********************************************************")
f.close()
