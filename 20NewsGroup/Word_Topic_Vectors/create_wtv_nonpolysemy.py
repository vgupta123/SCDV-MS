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
from scipy.sparse import hstack
from scipy.sparse.linalg import norm as normsp	
import os

WORD_EMBED_DIR = "../Word_Vectors"
INPUT_DIR = "../data/"


def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def cluster_GMM(num_clusters, word_vectors):
	# Initalize a GMM object and use it for clustering.
	clf =  GaussianMixture(n_components=num_clusters,
                    covariance_type="tied", init_params='kmeans', max_iter=50)
	# Get cluster assignments.
	clf.fit(word_vectors)
	idx = clf.predict(word_vectors)
	# print("Clustering Done...", time.time()-start, "seconds")
	# Get probabilities of cluster assignments.
	idx_proba = clf.predict_proba(word_vectors)
	# Dump cluster assignments and probability of cluster assignments. 
	joblib.dump(idx, 'gmm_latestclusmodel_len2alldata.pkl')
	# print("Cluster Assignments Saved...")

	joblib.dump(idx_proba, 'gmm_prob_latestclusmodel_len2alldata.pkl')
	# print("Probabilities of Cluster Assignments Saved...")
	return (idx, idx_proba)

def read_GMM(idx_name, idx_proba_name):
	# Loads cluster assignments and probability of cluster assignments. 
	idx = joblib.load(idx_name)
	idx_proba = joblib.load(idx_proba_name)
	# print("Cluster Model Loaded...")
	return (idx, idx_proba)

def get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict):
	# This function computes probability word-cluster vectors.
	
	prob_wordvecs = {}
	# for word in word_centroid_map:

	for word in word_centroid_map:
		prob_wordvecs[word] = np.zeros( num_clusters * num_features, dtype="float32" )
		# print(prob_wordvecs[word].shape)
		for index in range(0, num_clusters):
			try:
				prob_wordvecs[word][index*num_features:(index+1)*num_features] = word_vectors[word] * word_centroid_prob_map[word][index] * word_idf_dict[word]
			except:
				continue

	
	return prob_wordvecs

# def get_probability_word_vectors_sparseop(featurenames, word_centroid_map, num_clusters, word_idf_dict):
# 	# This function computes probability word-cluster vectors.
	
# 	prob_wordvecs = {}

# 	for word in word_centroid_map:
# 		prob_wordvecs[word] = sp(np.zeros( num_clusters * num_features, dtype="float32" ))
# 		l = []
# 		for index in range(0, num_clusters):
# 			try:
# 				l.append(word_vectors[word] * word_centroid_prob_map[word][index] * word_idf_dict[word])
# 			except:
# 				continue
# 		prob_wordvecs = hstack(l)
			
# 	return prob_wordvecs


filename = "scdv_sparsity_polysemy.txt"

num_features = int(sys.argv[1])
num_clusters = int(sys.argv[2])
choice = sys.argv[3]
sparsity = float(sys.argv[4])

f = open(filename,'a')

print("number of k clusters = " ,str(num_clusters))
print("dimension of word embeddings = " ,str(num_features))

f.write("number of k clusters " +str(num_clusters)+"\n")

start = time.time()

if choice == "doc2vecc":
	model_name = "Doc2VecC_non_Polysemy_"+str(num_features)+".txt"
	print("Embeddings = Doc2VecC")
elif choice == "word2vec":
	model_name = "Word2Vec_non_Polysemy_"+str(num_features)+".txt"
	print("Embeddings = Word2vec")

print("Sparsity = ",sparsity)
# if word2vec embeddings are used comment the below line else if doc2vec embedding are used comment line 99
word_vectors = KeyedVectors.load_word2vec_format(os.path.join(WORD_EMBED_DIR,model_name), binary=False)
# word_vectors = Word2Vec.load(os.path.join(WORD_EMBED_DIR,model_name)).wv

num_features = word_vectors.syn0.shape[1]

train = pd.read_csv(os.path.join(INPUT_DIR,'train_v2.tsv'), header=0, delimiter="\t")
# Load test data.
test = pd.read_csv(os.path.join(INPUT_DIR,'test_v2.tsv'), header=0, delimiter="\t")
all = pd.read_csv(os.path.join(INPUT_DIR,'all_v2.tsv'), header=0, delimiter="\t")


# Uncomment below line for creating new clusters.
start1 = time.time()

idx, idx_proba = cluster_GMM(num_clusters, word_vectors.syn0)
idx_proba[idx_proba<sparsity]=0

f.write("time taken in clustering " +str(time.time()-start1)+"\n")

# Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
# idx_name = "gmm_latestclusmodel_len2alldata.pkl"
# idx_proba_name = "gmm_prob_latestclusmodel_len2alldata.pkl"
# idx, idx_proba = read_GMM(idx_name, idx_proba_name)
# idx_proba[idx_proba<0.30]=0

# the below line may be uncommented to save the concatenation of word embeddings and the cluster probabilities 
# idx_proba = np.array(idx_proba)
# d = np.concatenate((word_vectors.syn0,idx_proba),axis=1)
# joblib.dump(dict(zip(word_vectors.index2word,d)),'concat_wordvectors_prob')
# print(d.shape)

# Create a Word / Index dictionary, mapping each vocabulary word to a cluster number
word_centroid_map = dict(zip( word_vectors.index2word, idx ))

# Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
# list of probabilities of cluster assignments.
word_centroid_prob_map = dict(zip( word_vectors.index2word, idx_proba ))

joblib.dump(word_centroid_prob_map,'word_centroid_prob_map')

# Computing tf-idf values.

traindata = []
for i in range( 0, len(all["news"])):
    traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(all["news"][i], True)))

start2 = time.time()

tfv = TfidfVectorizer(strip_accents='unicode',dtype=np.float32)
tfidfmatrix_traindata = tfv.fit_transform(traindata)
featurenames = tfv.get_feature_names()
idf = tfv._tfidf.idf_

# Creating a dictionary with word mapped to its idf value 
print("Creating word-idf dictionary for Training set...")

word_idf_dict = {}
for pair in zip(featurenames, idf):
    word_idf_dict[pair[0]] = pair[1]
    # for each_num in ["first","second","third","fourth","fifth"]:
    #     word_idf_dict[pair[0]+each_num] = pair[1]

start3 =time.time()
print("time taken in idf processing " +str(start3-start2)+"\n")

# Pre-computing probability word-cluster vectors.
wtv = get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict)

start4 =time.time()
print("time taken in Creating word topic Vectors " +str(start4-start3)+"\n")
temp_time = time.time() - start

joblib.dump(wtv,'wtv_nonpolysemy')

v = len(wtv)
v=0
c=0
for i in wtv:
    v+=1
    c+=np.count_nonzero(wtv[i])

print("vocab size = ",v)
print("total number of dimension of wtv vectors is ",num_clusters*num_features)
print("number of non zero entries in the wtv matrix is ",c)
print("percentage non zero entries in the wtv matrix = ",c/(v*num_clusters*num_features))

