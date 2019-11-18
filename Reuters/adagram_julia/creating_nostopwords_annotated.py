import pandas as pd
import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
import time
from sklearn.preprocessing import normalize
import sys
import pdb
import csv
import pickle
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
all = pd.read_pickle('all.pkl')
lb = MultiLabelBinarizer()
Y = lb.fit_transform(all.tags)
train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)

train = DataFrame({'text': []})
test = DataFrame({'text': []})

train["text"] = train_data.reset_index(drop=True)
test["text"] = test_data.reset_index(drop=True)
# Use the NLTK tokenizer to split the paragraph into sentences.
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
h = open("reuters_polysemy_text1.txt","r")
lines = h.readlines()
g = open("reuters_polysemy_text1_annotated.txt","r")
annotate_lines = g.readlines()
f = open("reuters_polysemy_text1_annotated_nostopwords.txt","w")
new_lines = [" ".join(line[:-1].split()) for line in lines]
# g = f.readlines()
# pdb.set_trace()
full_line = []
counter = 0
train_docs = []
for ind,review in enumerate(train["text"]):
    try:
        print ind
        wordvec_line = KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=False)
        doc_line = KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True)
        index_line = new_lines.index(" ".join(wordvec_line))
        annotate_line = annotate_lines[index_line].split()
        index_word = 0
        index_stop = 0
        annotate_word = []
        for each_word in wordvec_line:
            #pdb.set_trace()
            try:
                if(index_word<len(doc_line) and each_word==doc_line[index_word]):
                    annotate_word.append(annotate_line[index_stop])
                    index_word += 1
            except:
                pdb.set_trace()
            index_stop += 1
        # pdb.set_trace()
        train_docs.append(annotate_word)
    except:
        pdb.set_trace()
test_docs = []
for review in test["text"]:
        wordvec_line = KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=False)
        doc_line = KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True)
        index_line = new_lines.index(" ".join(wordvec_line))
        annotate_line = annotate_lines[index_line].split()
        index_word = 0
        index_stop = 0
        annotate_word = []
        for each_word in wordvec_line:
            #pdb.set_trace()
            if(index_word<len(doc_line) and each_word==doc_line[index_word]):
                annotate_word.append(annotate_line[index_stop])
                index_word += 1
            index_stop += 1
        # pdb.set_trace()
        test_docs.append(annotate_word)

train_text_annotate_sets = [train_docs,Y_train, test_docs,Y_test]
#pdb.set_trace()
pickle.dump(train_text_annotate_sets,open("train_text_annotate_sets.pkl","wb"))