function normalize_text {
  awk '{print tolower($0);}' < $1 | LC_ALL=C sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
  -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
}

echo "Enter the embeddings dimension"
read size

# shuffle the training set
shuf ../../data/20_newsgroup_sentences_1000.txt > ../../data/20_newsgroup_sentences_1000-shuf.txt
# cd ..

rm doc2vecc
gcc doc2vecc.c -o doc2vecc -lm -pthread -O3 -march=native -funroll-loops

# this script trains on all the data (train/test/unsup), you could also remove the test documents from the learning of word/document representation

time ./doc2vecc -train ../../data/20_newsgroup_sentences_1000-shuf.txt -word ../Doc2VecC_non_Polysemy_$size.txt -output docvectors_non_polysemy_$size.txt -cbow 1 -size $size -window 10 -negative 5 -hs 0 -sample 0 -threads 4 -binary 0 -iter 100 -min-count 10 -test ../../data/20_newsgroup_sentences_1000.txt -sentence-sample 0.1 -save-vocab alldata_non_polysemy_$size.vocab
