# Linear Classifier

The folder contains the code to run the linear SVM classifier on the Document Vector created.

Two possible options

## No reduction is carried out, then pass two agruements word2vec dimension and cluster
```sh
$ python3 SVM.py [dimension of word embedding] [number of clusters]
# eg - python3 SVM.py 200 60

$ python3 SVM.py [dimension of reduced of WTV]
# eg - python3 SVM.py 2000
``` 
	