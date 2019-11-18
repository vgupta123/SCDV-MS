import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.regularizers import l1
from keras.optimizers import Adam

from sklearn.externals import joblib
code_sizes= [500,1000,2000,3000]

for code_size in code_sizes:
    wtv = joblib.load('wtv_reuters_autoencoder')

    names=[]
    X=[]
    for i in wtv:
        names.append(i)
        X.append(wtv[i])
    X = np.array(X)

    m = np.max(X)
    X = X/m

    import pdb
    input_size = 12000

    input_img = Input(shape=(input_size,))
    hidden_3 = Dense(4000, activation='tanh')(input_img)
    code = Dense(code_size,activation='tanh')(hidden_3)
    hidden_4 = Dense(4000, activation='tanh')(code)
    output_img = Dense(input_size)(hidden_4)

    autoencoder = Model(input_img, output_img)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(X, X, epochs=5)

    X = np.matmul(X,autoencoder.get_weights()[0]) +autoencoder.get_weights()[1]
    X = np.tanh(X)
    X = np.matmul(X,autoencoder.get_weights()[2])+autoencoder.get_weights()[3]
    X = np.tanh(X)

    X = X*m

    file = open('dim.txt','w')
    file.write(str(X.shape))
    file.close()
    #pdb.set_trace()
    joblib.dump(dict(zip(names,X)),"new_wtv_"+str(code_size))