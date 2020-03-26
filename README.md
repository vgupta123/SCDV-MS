# Improving Document Classification with Multi-Sense Embeddings


## Introduction
  - For text classification and information retrieval tasks, text data has to be represented as a fixed dimension vector. 
  - We propose important modification to the simple feature construction technique named **Sparse Composite Document Vector (SCDV)** [1]
  - We demonstrate our method through experiments on multi-class classification on 20newsgroup dataset and multi-label text classification on Reuters-21578 dataset. 

## Testing
There are 2 folders named 20news and Reuters which contains code related to multi-class classification on 20Newsgroup dataset and multi-label classification on Reuters dataset.

#### 20Newsgroup
Change directory to 20news for experimenting on 20Newsgroup dataset and all train and test data files (both annotated and non-annoated with adagram-julia are there) as follows:
```sh
$ cd 20NewsGroup
```
Get word vectors (with Doc2VecC on polysemous corpus) for all words in vocabulary (you can also create other word embedding by following local readme):
```sh
$ cd Word_Vectors
$ cd doc2vecC
$ sh go_polysemy_20news_polysemy.sh 
$ cd ../
# Word2Vec.py takes word vector dimension as an argument. you can input a dimension of 200.
```

Get word topic vectors for all the word vectors and sparse GMM (read local readMe for more details):
```sh
$ cd Word_Topic_Vectors
$ python3 create_wtv.py 200 60 doc2vecc 0.3
$ cd ../
# create_wtv.py takes word vector dimension, number of clusters as arguments, type of embeddings and sparsity threshold. We took word vector dim 200, 60 as number of clusters, doc2vecc train word-vectors and sparsity threshold of 0.3
```

Get Sparse Document Vectors (SCDV) for documents in train and test set and accuracy of prediction on test set (read local readMe for more details):
```sh
$ cd SVM_classifier
$ python SVM.py 200 60
$ cd ../
# SCDV.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```
#### Reuters
Change directory to Reuters for experimenting on Reuters-21578 dataset. Similar to 20newsgroup folder, read local readMe for more details:

## Requirements
Minimum requirements:
  -  Python 2.7+
  -  NumPy 1.8+
  -  Scikit-learn
  -  Pandas
  -  Gensim

## References
[1] Mekala, Dheeraj, Vivek Gupta, Bhargavi Paranjape, and Harish Karnick. "SCDV: Sparse Composite Document Vectors using soft clustering over distributional representations." In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 659-669. 2017

    Note: You neednot download 20Newsgroup or Reuters-21578 dataset. All datasets are present in their respective directories.

[//]: # (We used SGMl parser for parsing Reuters-21578 dataset from  https://gist.github.com/herrfz/7967781)

