import numpy as np
import pandas as pd
import math

codeLength = 16
lambd = 1.0
v_constant = 1e-5
counter = 6

#Initializing the B matrix, ytrain, phi_train
B = pd.DataFrame(np.random.randint(1,3,size = (codeLength,60000)))
B[B == 2] = -1
ytrain = pd.read_pickle("Dataset/labels_train.pkl")
phi_train = pd.read_pickle("Dataset/phi_images_train.pkl")
y = np.zeros(shape=(len(ytrain.index),10))
y = pd.DataFrame(y)
for i in ytrain.index:
	y.loc[i,ytrain.loc[i,0]] = 1

ytrain = y
#Initializing the B matrix
B = pd.DataFrame(np.random.randint(1,3,size = (codeLength,len(ytrain.index))))
B[B == 2] = -1

while counter > 1:
	#G Step Computing W
	W = pd.DataFrame(np.dot(B,B.transpose()))
	for i in range(1,len(W)):
		W.iloc[i][i] = W.iloc[i][i] + lambd
	df_inv = pd.DataFrame(np.linalg.pinv(W.values), W.columns, W.index)
	W = pd.DataFrame(np.dot(np.dot(df_inv,B),ytrain))
	#F Step Computing P
	P = np.dot(np.linalg.pinv(np.dot(phi_train.transpose(),phi_train)),np.dot(phi_train.transpose(),B.transpose()))
	P = pd.DataFrame(P)
	
	#B Step Computing B iteratively	
	Q = np.dot(W,ytrain.transpose()) +  v_constant * np.dot(P.transpose(),phi_train.transpose())
	Bold = B
	for i in range(1,codeLength):
		remove_index = B.index.isin([i,])
		Bd = B[~remove_index]
		remove_index = W.index.isin([i,])
		Wd = W[~remove_index]
		v = W[remove_index]
		q = Q[i:i+1,:]
		z = np.sign(q.transpose() - np.dot(np.dot(Bd.transpose(),Wd),v.transpose()))
		B.loc[i:i,:] = z.transpose()
	
	#Computing l2 loss
	'''
	temp = 0
	temp = np.dot(B.transpose(),Q)
	temp = np.linalg.norm(temp,'fro')
	l2 = math.pow(np.linalg.norm(np.dot(W.transpose(),B)),2) - 2*temp
	print l2
	'''	
	counter = counter-1	
	print counter
	if np.linalg.norm(B-Bold) < 1e-6 * np.linalg.norm(Bold) or counter < 2:
		W = pd.DataFrame(np.dot(B,B.transpose()))
		for i in range(1,len(W)):
			W.iloc[i][i] = W.iloc[i][i] + lambd
		df_inv = pd.DataFrame(np.linalg.pinv(W.values), W.columns, W.index)
		W = pd.DataFrame(np.dot(np.dot(df_inv,B),ytrain))
		P = np.dot(np.linalg.pinv(np.dot(phi_train.transpose(),phi_train)),np.dot(phi_train.transpose(),B.transpose()))
		P = pd.DataFrame(P)
		break

W.to_pickle("Dataset/W.pkl")
P.to_pickle("Dataset/P.pkl")
B.to_pickle("Dataset/B.pkl")

'''
CodeLength 16
real	8m18.353s
user	12m34.132s
sys	0m48.644s

CodeLength 32
real	17m24.318s
user	22m6.836s
sys	0m14.912s

CodeLength 64
real	36m38.497s
user	40m24.960s
sys	1m7.732s

CodeLength 96
real	47m36.725s
user	51m51.720s
sys	1m24.004s

CodeLength 128
real	63m33.906s
user	68m15.808s
sys	0m50.496s
'''
