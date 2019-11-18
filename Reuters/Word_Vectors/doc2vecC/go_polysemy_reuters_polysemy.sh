function normalize_text {
  awk '{print tolower($0);}' < $1 | LC_ALL=C sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
  -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
}

#wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
#tar -xf aclImdb_v1.tar.gz
## normalize the data
cd ../../data
'''
for j in train/pos train/neg test/pos test/neg train/unsup; do
  rm temp
  rm $j/norm.txt
  for i in `ls $j`; do cat $j/$i >> temp; awk 'BEGIN{print;}' >> temp; done
  normalize_text temp
  mv temp-norm $j/norm.txt
done
'''
#cat train/pos/norm.txt train/neg/norm.txt train/unsup/norm.txt test/pos/norm.txt test/neg/norm.txt > alldata.txt
## shuffle the training set
shuf reuters_polysemy_text1_annotated.txt > reuters_polysemy_text1_annotated-shuf.txt
cd ../Word_Vectors/doc2vecC

rm doc2vecc
gcc doc2vecc.c -o doc2vecc -lm -pthread -O3 -march=native -funroll-loops

# this script trains on all the data (train/test/unsup), you could also remove the test documents from the learning of word/document representation
time ./doc2vecc -train ../../data/reuters_polysemy_text1_annotated-shuf.txt -word wordvectors_reuters_polysemy.txt -output ../../data/docvectors_polysemy.txt -cbow 1 -size 200 -window 10 -negative 5 -hs 0 -sample 0 -threads 4 -binary 0 -iter 100 -min-count 10 -test ../../data/reuters_polysemy_text1_annotated.txt -sentence-sample 0.1 -save-vocab ../../data/alldata_polysemy.vocab

#head -n 25000 docvectors.txt | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<=NF; b++) printf b ":" $(b) " "; print ""; a++;}' > train.txt
#tail -n 25000 docvectors.txt | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<=NF; b++) printf b ":" $(b) " "; print ""; a++;}' > test.txt
#PATH_TO_LIBLINEAR/train -s 0 train.txt model.logreg
#PATH_TO_LIBLINEAR/predict -b 1 test.txt model.logreg out.logreg
#cd ..
