import pdb
import numpy as np

f = open("20_newsgroup_tokenized_multisense_docs_1000_test.txt","r")
g = open("20_newsgroup_tokenized_multisense_docs_1000_train.txt","r")
lines = f.readlines()+g.readlines()
words = []
for each_line in lines:
    words.extend(each_line.split())

words = list(set(words))

poly_list_words = []
for each_word in words:
    if(any([1 for key in ["first","second","third","fourth","fifth"] if(key in each_word)])):
        poly_list_words.append(each_word)


#for

poly_new_words = []
list_key = ["first","second","third","fourth","fifth"]
for each_word in poly_list_words:
    for each_key in list_key:
        if(each_key in each_word):
            if(each_word.replace(each_key, "")+"first" in poly_list_words):
                poly_new_words.append(each_word.replace(each_key,""))
#pdb.set_trace()
poly_new_words = list(set(poly_new_words))
fil_write = open("20newsgroup_words.txt","w")
for poly_word in poly_new_words:
    fil_write.write(poly_word+"\n")