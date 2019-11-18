# Adagram
The Folder contains the necessary files to generate the word embeddings. Both of the scipts will store the embeddings in the folder Word_Vectors

Two methods to generate embeddings have been used 

## Word2Vec (gensim) : 

```sh
$ python Word2Vec.py
# for embeddings of 200 dimension on single-sense annotated corpus
$ python Word2Vec_multisense.py
# for embeddings of 200 dimension on multi-sense annotated corpus
``` 

## Doc2VecC : 

Change the directory to the Doc2VecC folder. 
```sh
$ cd doc2vecC
# change directoy to doc2vecC
```

Run the bash scripts to get desired embeddings
```sh
$ bash go_polysemy_reuters_nonpoly.sh
# for embeddings of 200 dimension on single-sense annotated corpus
$ bash go_polysemy_reuters_polysemy.sh
# for embeddings of 200 dimension on multi-sense annotated corpus
```