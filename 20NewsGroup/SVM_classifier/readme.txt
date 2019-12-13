the folder contains the code to run the linear SVM classifier on the Word topic vectors created

Files :

SVM.py   			: To run Classifier on polysemy corpus
SVM_nonpolysemy.py 	: To run Classifier on non polysemy corpus	


How to run ?

	the script can be run in two ways ?

	Case 1 : if no reduction is carried out , then pass two agruements 

			python3 SVM.py [dimension of word embedding] [number of clusters]
			eg - python3 SVM.py 200 60

			python3 SVM_nonpolysemy.py [dimension of word embedding] [number of clusters]
			eg - python3 SVM_nonpolysemy.py 200 60

	Case 2 : if reduction is carried out , then pass one agruement i.e the final dimension of the WTV
	
			python3 SVM.py [dimension of reduced of WTV]
			eg - python3 SVM.py 2000

			python3 SVM_nonpolysemy.py [dimension of reduced of WTV]
			eg - python3 SVM_nonpolysemy.py 2000
