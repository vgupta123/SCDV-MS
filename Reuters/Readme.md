The following folder contains file to run SCDV with polysemy corpus

Directory Structure :

1. adagram_julia : Contains the code to create dataset and annotated files.

2 . Word_Vectors : Contains the code to create word embeddings

3 . Word_Topic_Vecotrs : Contains the code to create word topic vectors with redcution code

4 . data : contains the Reuters data along with polysemy word corpus

5 . SVM_classifier : Code to run the linear SVM on the generated embeddings

6 . Metrics : code to generate the scores for reuters data


Run the codes in the following order :

adagram_julia -> Word_vectors -> Word_topic_Vectors
SVM_classier -> Metrics
then either SVM_classier or CNN .