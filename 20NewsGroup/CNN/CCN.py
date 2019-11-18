# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# Any results you write to the current directory are saved as output.

# from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D , GlobalMaxPooling2D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Conv2D,Dropout
from keras.models import Model
from keras.regularizers import l2
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


BASE_DIR = ''
GLOVE_DIR = './'
TEXT_DATA_DIR = '../data'
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = int(input("Enter the embedding dimensions : "))
VALIDATION_SPLIT = 0.2


print('Indexing word vectors.')
file_name = input("Enter the file name : ") 	
embeddings_index = {}

with open(os.path.join(GLOVE_DIR, file_name)) as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        if coefs.shape!=(EMBEDDING_DIM,):
        	continue
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

def load_data(TEXT_DATA_DIR):
   
    print('Processing text dataset')
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    # for name in sorted(os.listdir(TEXT_DATA_DIR)):
    #     path = os.path.join(TEXT_DATA_DIR, name)
    #     if os.path.isdir(path):
    #         label_id = len(labels_index)
    #         labels_index[name] = label_id
    #         for fname in sorted(os.listdir(path)):
    #             if fname.isdigit():
    #                 fpath = os.path.join(path, fname)
    #                 if sys.version_info < (3,):
    #                     f = open(fpath)
    #                 else:
    #                     f = open(fpath, encoding='latin-1')
    #                 texts.append(f.read())
    #                 f.close()
    #                 labels.append(label_id)

    train = pd.read_csv(os.path.join(TEXT_DATA_DIR,'train_v2.tsv'), header=0, delimiter="\t")
    # Load test data.
    test = pd.read_csv(os.path.join(TEXT_DATA_DIR,'test_v2.tsv'), header=0, delimiter="\t")
    all = pd.read_csv(os.path.join(TEXT_DATA_DIR,'all_v2.tsv'), header=0, delimiter="\t")

    l1 = list(train['class'])
    labels = l1
    l2 = list(test['class'])
    labels = l1 + l2

    e = open(os.path.join(TEXT_DATA_DIR,'20_newsgroup_tokenized_multisense_sentences_1000_train.txt'),"r")
    g = open(os.path.join(TEXT_DATA_DIR,'20_newsgroup_tokenized_multisense_sentences_1000_test.txt'),"r")
    train_lines = e.readlines()
    train_lines_mod = [line[:-1].split() for line in train_lines]
    test_lines = g.readlines()
    test_lines_mod = [line[:-1].split() for line in test_lines]
    texts = train_lines + test_lines
    # texts = train_lines

    print('Found %s texts.' % len(texts))
    print("len labes = ",len(labels))
    global word_index, tokenizer

    tokenizer = Tokenizer(nb_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    e.close()
    g.close()


    data_train = data[:len(train_lines)]
    data_test = data[-len(train_lines):]
    labels_train = labels[:len(l1)]
    labels_test = labels[-len(l1):]

    return (data_train, labels_train, data_test,labels_test)


data , labels , x_test , y_test = load_data(TEXT_DATA_DIR)  # list of text samples

print('Found %s texts.' % len(data))


# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)

# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

# data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)



indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

from keras.callbacks import ModelCheckpoint



print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='tanh')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Dropout(0.4, noise_shape=None, seed=None)(x)
x = Conv1D(128, 5, activation='tanh')(x)
x = MaxPooling1D(5)(x)
x = Dropout(0.4, noise_shape=None, seed=None)(x)
x = Conv1D(128, 5, activation='tanh')(x)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.4, noise_shape=None, seed=None)(x)
x = Dense(128, activation='tanh')(x)
preds = Dense(20, activation='softmax')(x)

model = Model(sequence_input, preds)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model.fit(x_train, y_train,
          batch_size=20,
          epochs=20,
          validation_data=(x_val, y_val),
          callbacks=callbacks_list)


model.load_weights("weights.best.hdf5")
y_predict = model.predict(x_test)




for i in range(len(y_predict)):
   y_predict[i][y_predict[i]==np.max(y_predict[i])]=1
   y_predict[i][y_predict[i]!=1]=0 

print(classification_report(y_test,y_predict,digits=5))


print("Test accuracy = ",accuracy_score(y_test,y_predict))

