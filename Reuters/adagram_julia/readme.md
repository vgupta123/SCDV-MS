Change directory to adagram_julia for creating the annotated Reuters-21578 dataset

As reuters data is in SGML format, parsing data and creating pickle file of parsed data can be done as follows:
```sh
$ python create_data.py
# We don't save train and test files locally. We split data into train and test whenever needed.
```
extract raw text from the data ignoring labels :
```sh
$ python extract_text.py
# output raw sentences in reuters_polysemy_text1.txt
```
Setup the AdaGram algorithm for multi-sence corpus annotation
```sh
$ ~/.julia/v0.4/AdaGram/utils/tokenize.sh reuters_polysemy_text1.txt reuters_tokenized.txt
# output tokenized sentences from raw text
$ ~/.julia/v0.4/AdaGram/utils/dictionary.sh reuters_tokenized.txt reuters_dict.txt
# output dictionary of words in the corpus
$ ~/.julia/v0.4/AdaGram/train.sh --min-freq 1 reuters_tokenized.txt reuters_dict.txt reuters_model_multisense
# output multsense word embeeding. we took min word frequency =1 for our experiment
$ ~/.julia/v0.4/AdaGram/build.sh
# build the adagram model files
$ ~/.julia/v0.4/AdaGram/run.sh first.jl 
# run the adagram model
```
Use the AdaGram model for corpus annotation
```sh
$ python analyzing_multisense_20news.py
```
Commands to move annotated data to data folder
```sh
$ cp reuters_polysemy_text2_annotated.txt ../data/reuters_polysemy_text2_annotated.txt
$ cp reuters_polysemy_text1.txt ../data/reuters_polysemy_text1.txt
$ cp reuters_polysemy_text1_annotated.txt ../data/reuters_polysemy_text1_annotated.txt
$ cp train_text_annotate_sets.pkl ../data/train_text_annotate_sets.pkl
```
