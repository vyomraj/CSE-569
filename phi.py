import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

xtrain = pd.read_pickle("Dataset/images_train.pkl")
ytrain = pd.read_pickle("Dataset/labels_train.pkl")
xtest = pd.read_pickle("Dataset/images_test.pkl")
ytest = pd.read_pickle("Dataset/labels_test.pkl")

#Initialize sigma and anchor
sigma = 0.4
anchor_count = 1000
anchor =  xtrain.sample(anchor_count)
anchor.to_pickle("Dataset/anchor.pkl")

#Creating Phi matrix
phi_xtrain = pd.DataFrame(0, index = range(len(xtrain.index)), columns = range(anchor_count))
phi_xtrain = np.exp(pow(euclidean_distances(xtrain,anchor),2)/(-2*sigma*sigma))
phi_xtrain = pd.DataFrame(phi_xtrain)
phi_xtrain.to_pickle("Dataset/phi_images_train.pkl")

phi_xtest = pd.DataFrame(0, index = range(len(xtest.index)), columns = range(anchor_count))
phi_xtest = np.exp(pow(euclidean_distances(xtest,anchor),2)/(-2*sigma*sigma))
phi_xtest = pd.DataFrame(phi_xtest)
phi_xtest.to_pickle("Dataset/phi_images_test.pkl")
