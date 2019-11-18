#!/usr/bin/env python
import pandas as pd
import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
import time
from sklearn.preprocessing import normalize
import sys
import csv
import nltk
import os
nltk.download('punkt')

INPUT_DIR = "../data/"

if __name__ == '__main__':

	
	num_features = int(sys.argv[1])   # Word vector dimensionality
	min_word_count = 20   # Minimum word count
	num_workers = 4       # Number of threads to run in parallel
	context = 10          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words

	start = time.time()
	# The csv file might contain very huge fields, therefore set the field_size_limit to maximum.
	csv.field_size_limit(sys.maxsize)

	
	# Read train data for polysemy corpus .
	# file = open(os.path.join(INPUT_DIR,'20_newsgroup_tokenized_multisense_sentences_1000.txt'),'r')
	# train_word_vector = file.readlines()
	train_word_vector = pd.read_csv("../data/all_v2.tsv",sep='\t')
	# print(train_word_vector)

	# Use the NLTK tokenizer to split the paragraph into sentences.
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = []

	print("Parsing sentences from training set...")	
	# print(len(train_word_vector))
	# Loop over each news article.
	for review in train_word_vector["news"]:
		# print(1)
		try:
			# Split a review into parsed sentences.
			sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
		except:
			continue

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

	
	print("Training Word2Vec model...")
	# Train Word2Vec model.

	# print(len(sentences))
	# print(len(sentences[0]))
	# print(sentences[0])
	
	model = Word2Vec(sentences, workers=num_workers, hs = 1, sg = 1, negative = 5, iter = 25,\
	            size=num_features, min_count = 10, \
	            window = 10, sample = 0, seed=1)
	
	
	model_name = "Word2Vec_"+str(num_features)+"Dim"

	model.init_sims(replace=True)
	
	# Save Word2Vec model.
	print("Saving Word2Vec model...")	
	# model.save(model_name)
	model.wv.save_word2vec_format("Word2Vec_non_Polysemy_"+str(num_features)+".txt", binary=False)
	endmodeltime = time.time()
	print("time : ", endmodeltime-start)
