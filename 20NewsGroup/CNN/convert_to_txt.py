from sklearn.externals import joblib
from gensim.models import Word2Vec

file = input("Enter the file name : ")
wtv = joblib.load(file)

for i in wtv:
	dim = wtv[i].shape[0]
	break
print("Dimension of embedding : ",dim)
file = open(file+"_txt.txt",'w')
for i in wtv :
	file.write(i+" ")
	for j in range(dim):
		file.write(str(wtv[i][j])+" ")
	file.write("\n")
file.close() 
