from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import time
import sys
from sklearn.externals import joblib
import gc
import pickle
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import sparse
# from postProcsess import processVecMatrix

def pca(X,final_dimension):

	X = np.array(X)
	X = X - np.average(X,axis=0)
	sklearn_pca = PCA(n_components=final_dimension,svd_solver='full')
	# X_pca = sklearn_pca.fit(X)
	X_pca = sklearn_pca.fit_transform(X)
	return X_pca

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def random_proj_sparse_random(X,n_comp):

    rp = random_projection.SparseRandomProjection(n_components=n_comp, random_state=42)
    X_projected = rp.fit_transform(X)
    del rp
    return X_projected

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def random_proj_gaussian_random(X,n_comp):

    rp = random_projection.GaussianRandomProjection(n_components=n_comp, random_state=42)
    X_projected = rp.fit_transform(X)
    del rp
    return X_projected


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

def clusters(X):

	clusters = [X[:,i:i+200] for i in range(0,12000,200)]
	modified_clusters = []
	for cluster in clusters:
		rank = np.linalg.matrix_rank(cluster)
		if rank>100:        
			pca = PCA(n_components=50,svd_solver='full')
			modified_clusters.append(pca.fit_transform(cluster))
		else:
			pca = PCA(n_components=rank,svd_solver='full')
			modified_clusters.append(pca.fit_transform(cluster))
	new_wtv = np.concatenate(modified_clusters,axis=1)
	return new_wtv

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


num_features = int(sys.argv[1])
num_clusters = int(sys.argv[2])
final_dimension = int(sys.argv[3])

# Loading Probability Word vector dictionary

name = "../wtv"
start = time.time()

prob_wordvecs = joblib.load(name)
temp_time = time.time() - start
print("WTV (Dictionary ) Loaded ...", temp_time, " seconds")

X = []
names = []
for i in prob_wordvecs:
    X.append(prob_wordvecs[i])
    names.append(i)

X = np.array(X)

print(X.shape)
print(len(names))

del prob_wordvecs
gc.collect()

print("average = ",np.linalg.norm(np.average(X,axis=0)))

#standardize the data
# sc = StandardScaler()
# X = sc.fit_transform(X)
# print("average = ",np.linalg.norm(np.average(X,axis=0)))
# # print("normalizing topic vectors before reduction...")
# # X = normalize(X)
# avg = 0
# for i in range(len(X)):
# 	avg+=np.linalg.norm(X[i])
# avg=avg/len(X)
# print("Average of norms of topic vectors before reduction = ",avg)

print("starting reduction...")
start = time.time()

print("Enter a option to select the reduction option ")
print("1 . PCA")
print("2 . Random Projection (Gaussian)")
print("3 . Apply PCA on individual clusters")
choice = int(input("Enter the choice : "))

if choice == 1 :
	new = pca(X,final_dimension)
elif choice == 2 :
	new = random_proj_gaussian_random(X,final_dimension)
elif choice == 3:
	new = clusters(X)

print("finished reduction in ",time.time()-start," seconds ...")
# print("post prob_wordvecscessing")
# new = processVecMatrix(new,10)
# print("done")

# print("normalizing topic vectors after reduction...")
# new = normalize(new)

# avg = 0
# for i in range(len(new)):
# 	avg+=np.linalg.norm(new[i])
# avg=avg/len(new)

# print("Average of norms of topic vectors after reduction = ",avg)

# print("sparsing the reduced vectors")
# ans1 = np.sum(np.min(new,axis=1))
# ans2 = np.sum(np.max(new,axis=1))
# ans = (abs(ans1)+abs(ans2))/2
# ans=ans/len(new)
# threshold = ans*0.03
# index = np.abs(new)<threshold
# new[index]=0

print("shape of the fit_transformed data is ",new.shape)

joblib.dump(dict(zip(names, new)), '../reduced_wtv')
temp_time = time.time() - start
print("Completed at ", temp_time, " seconds")
