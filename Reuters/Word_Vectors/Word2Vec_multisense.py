#!/usr/bin/env python
import pandas as pd
import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
#from KaggleWord2VecUtility import KaggleWord2VecUtility
import time
from sklearn.preprocessing import normalize
import sys
import pdb
import csv

if __name__ == '__main__':
      #print sys.maxsize
      start = time.time()
	# The csv file might contain very huge fields, therefore set the field_size_limit to maximum.
      csv.field_size_limit(10000000)
	# Read train data.
      #train_word_vector = pd.read_csv( 'data/all_v2.tsv', header=0, delimiter="\t")
	# Use the NLTK tokenizer to split the paragraph into sentences.
      tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
      sentences = []
      import pdb

      f = open("../data/reuters_polysemy_text2_annotated.txt", "r")
      file_read = f.readlines()
      print len(file_read)
      sentences = []
      for each_line in file_read:
            sentences.append(each_line[:-1].split())
      #pdb.set_trace()
      logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
      num_features = int(sys.argv[1])     # Word vector dimensionality
      #for min_word in [3,6]:
      min_word_count = 20   # Minimum word count
      num_workers = 40       # Number of threads to run in parallel
      context = 10          # Context window size
      downsampling = 1e-3   # Downsample setting for frequent words
      print "Training Word2Vec model..."
    # Train Word2Vec model.
      print len(sentences)
      model = Word2Vec(sentences, workers=num_workers, hs = 0, sg = 1, negative = 10, iter = 25,\
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)
      model_name = "Word2Vec_Polysemy_"+str(num_features)+"Dim"
      #model.init_sims(replace=True)
      model.save(model_name)
      model = Word2Vec.load(model_name)
      word_vectors = Word2Vec.load(model_name).wv.syn0
      print len(word_vectors)
      print len(word_vectors[0])
      print word_vectors.shape
      print model.wv.index2word
    # Save Word2Vec model.
      print "Saving Word2Vec model..."
      #model.save(model_name)
      endmodeltime = time.time()
      print "time : ", endmodeltime-start
