from mnist import MNIST
import pandas as pd
import pickle

mndata = MNIST('./Dataset')
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()

images_train = [[float(i)/255 for i in j] for j in images_train]
images_test  = [[float(i)/255 for i in j] for j in images_test]

df_images_train = pd.DataFrame(images_train)
df_labels_train = pd.DataFrame(list(labels_train))
df_images_test = pd.DataFrame(images_test)
df_labels_test = pd.DataFrame(list(labels_test))

df_images_train.to_pickle("Dataset/images_train.pkl")
df_labels_train.to_pickle("Dataset/labels_train.pkl")
df_images_test.to_pickle("Dataset/images_test.pkl")
df_labels_test.to_pickle("Dataset/labels_test.pkl")
