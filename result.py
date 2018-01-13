import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import scipy as sp
from sklearn.metrics import average_precision_score

ytrain = pd.read_pickle("Dataset/labels_train.pkl")
W = pd.read_pickle("Dataset/W.pkl")
P = pd.read_pickle("Dataset/P.pkl")
Btrain = pd.read_pickle("Dataset/B.pkl")
ytraind = pd.DataFrame(np.dot(W.transpose(),Btrain))
ytraind = pd.DataFrame(ytraind.transpose().idxmax(axis = 1))
losstrain = np.absolute(ytrain - ytraind)

print "Training Set Accuracy"
print (len(ytrain.index) - np.count_nonzero(losstrain)) *1.0 / len(ytrain.index)

ytest = pd.read_pickle("Dataset/labels_test.pkl")
phi_test = pd.read_pickle("Dataset/phi_images_test.pkl")

Btest = np.sign(np.dot(P.transpose(),phi_test.transpose()))
Btest = pd.DataFrame(Btest)
ytestd = pd.DataFrame(np.dot(W.transpose(),Btest))
ytestd = pd.DataFrame(ytestd.transpose().idxmax(axis = 1))
losstest = np.absolute(ytest - ytestd)

print "Test Set Accuracy"
print (len(ytest.index) - np.count_nonzero(losstest)) * 1.0 / len(ytest.index)

hammDistTest = pd.DataFrame(pairwise_distances(Btest.transpose(), metric = "hamming"),index = Btest.columns, columns = Btest.columns)
hammDistTest = hammDistTest * len(Btest.index)

precision = 0.0
recall = 0.0

for i in hammDistTest.columns:
	hammDistRow = pd.DataFrame(hammDistTest.loc[i])
	retSet = hammDistRow[i] < 2.00001
	retSet = ytest[retSet]
	retSet[0] =  ytest.loc[i,0]
	trueSet = ytest[0] == ytest.loc[i,0]
	trueSet = ytest[trueSet]
	
	precision = precision + (len(list(set(retSet.index) & set(trueSet.index))) *1.0/ len(retSet.index))
	recall = recall + (len(list(set(retSet.index) & set(trueSet.index))) *1.0/len(trueSet))
	'''tp = 0
	for j in hammDistRow[retSet].index:
		if ytest.loc[j][0] == ytest.loc[i][0]:
			tp = tp +1
	precision = precision + tp*1.0/len(hammDistRow[retSet])
	recall = recall +  tp*1.0/len(ytest[ytest[0] == ytest.loc[i][0]])'''

precision = precision / len(Btest.columns)
recall = recall / len(Btest.columns)


print "Precision"
print precision
print "Recall"
print recall

'''
Code Length 16
Training Set Accuracy
1.0
Test Set Accuracy
0.8932
Precision
0.803321771943
Recall
0.805216368164

real	3m17.500s
user	3m17.472s
sys	0m0.856s

Code Length 32	
Training Set Accuracy
1.0
Test Set Accuracy
0.893
Precision
0.80349386489
Recall
0.779278719886

real	2m53.138s
user	2m53.396s
sys	0m0.892s

Code Length 64
Training Set Accuracy
1.0
Test Set Accuracy
0.8931
Precision
0.803747264997
Recall
0.757152486973

real	3m13.013s
user	3m13.152s
sys	0m0.952s

Code Length 96
Training Set Accuracy
1.0
Test Set Accuracy
0.8932
Precision
0.805671579596
Recall
0.575059136481

real	3m5.349s
user	3m5.740s
sys	0m0.920s

Code Length 128
Training Set Accuracy
1.0
Test Set Accuracy
0.8929
Precision
0.805323613047
Recall
0.42521599734

real	2m55.999s
user	2m56.404s
sys	0m1.024s
'''
