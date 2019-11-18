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
import pdb
if __name__ == '__main__':

	start = time.time()
	# The csv file might contain very huge fields, therefore set the field_size_limit to maximum.
	csv.field_size_limit(sys.maxsize)
	# Read train data.
	train_word_vector = pd.read_pickle('all.pkl')
	# Use the NLTK tokenizer to split the paragraph into sentences.
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = []
	print "Parsing sentences from training set..."

	# Loop over each news article.
    	list_sen = []
	f = open("reuters_polysemy_text2.txt", "w")
	g = open("reuters_polysemy_text1.txt", "w")
	f.close()
	g.close()
	f = open("reuters_polysemy_text2.txt", "a")
	g = open("reuters_polysemy_text1.txt", "a")
	for i,review in enumerate(train_word_vector["text"]):
			# Split a review into parsed sentences.
			sentences = KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
			#print i
            		#print sentences
			for sen in sentences:
				f.write(" ".join(sen)+ "\n")
			g.write(" ".join([" ".join(listi) for listi in sentences])+"\n")
			list_sen.append(" ".join([" ".join(listi) for listi in sentences]))
	#pdb.set_trace()
	g = open("reuters_polysemy_text.txt", "w")
	g.write(" ".join(list_sen))


