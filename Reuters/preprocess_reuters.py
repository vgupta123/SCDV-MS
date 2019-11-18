import pdb
import pickle
import numpy as np

fil = open("reuters_tokenized_multisense_docs_250.txt","w")
train_docs, Y_train, test_docs, Y_test = pickle.load(open("train_text_annotate_sets.pkl","r"))
lines = train_docs+test_docs
words = []
for each_line in lines:
    fil.write(" ".join(each_line)+"\n")