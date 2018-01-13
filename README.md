The project is to implement Supervised Discrete Hashing algorithm on MNIST dataset
and calculate precision and recall on the same.

System Requirements
This was tested on sytem with Intel Core I5, 8GB Ram, 64 bit Ubuntu 16.04 LTS
Python 2.7 was installed on system with required libraries

Download link MNIST Dataset
http://yann.lecun.com/exdb/mnist/

Run the following code - to convert mnist to pkl file
python mnist_load.py

Run the following code - to create phi matrix for test and training data(you can edit the no of anchor points here)
python phi.py

Run the following code - to run SDH and save B,P,W matrix (you can edit the codeLength here)
python SDH.py

Run the following code - to collect statistics precision, recall and accuracy
python -W ignore result.py

Conditions
Folder Dataset should contain (Unzip Dataset.zip)
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
train-images-idx3-ubyte
train-labels-idx1-ubyte
Please ignore the warning SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.

Run information of SDH.py and result.py is given in the python code as comments
Dependency list is in file 'Dependecy List'
Citations are given in file 'Citations'

Single File to run entire code for a code length is created
It is run.py
python -W ignore run.py
This displays graph results of results we have stored(it displays current result in console)

This software is available under the following licenses:
	MIT

Link for complete code with dataset
https://www.dropbox.com/s/npo4qqp90oyvtmi/FSL.zip?dl=0

