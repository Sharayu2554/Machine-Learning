import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))

if(len(sys.argv) != 3):
	print("Command is Python Q1_Census_Income.py __INPUT_TRAIN_FILE_PATH__ __INPUT_TEST_FILE_PATH__ ")
	print("Example : Python Q1_Census_Income.py '../dataset/adultTrain.data' '../dataset/adultTest.data' ")

dataset = pd.read_csv(sys.argv[1], sep=',', encoding='latin1', header= None)
testset = pd.read_csv(sys.argv[2], sep=',', encoding='latin1', header= None)

le=LabelEncoder()
tle=LabelEncoder()
for column in dataset:
    if dataset[column].dtypes == 'object':
        data=dataset[column]
        le.fit(data.values)
        dataset[column] = le.transform(dataset[column])
        
for column in testset:
    if testset[column].dtypes == 'object':
        testdata=testset[column]
        tle.fit(testdata.values)
        testset[column] = tle.transform(testset[column])

X_train = dataset.values[:, 0:14]
y_train = dataset.values[:, -1]

X_test = testset.values[:, 0:14]
y_test = testset.values[:, -1]

print("\n*** Running NN Model ****\n")
mlp = MLPClassifier(hidden_layer_sizes=(23,23,23), max_iter=1000)
mlp.fit(X_train,y_train)

predictions_train = mlp.predict(X_train)
print("\nConfusion Matrix of NN Training set : ")
print(confusion_matrix(y_train,predictions_train))

predictions_test = mlp.predict(X_test)
print("\nConfusion Matrix of NN Test set : ")
print( confusion_matrix(y_test,predictions_test))

accuracy = sum(1 for i in range(len(predictions_train)) if predictions_train[i] == y_train[i]) / float(len(predictions_train))
print("\nAccuracy of NN Training set : {0:.4f}".format(accuracy))

accuracy = sum(1 for i in range(len(predictions_test)) if predictions_test[i] == y_test[i]) / float(len(predictions_test))
print("\nAccuracy of NN Test set : {0:.4f}".format(accuracy))

svc_radial = svm.SVC()
print("\n*** Running SVM Model ****\n")
print("*** This might take a while ***")
svc_radial.fit(X_train, y_train)
predicted_train= svc_radial.predict(X_train)
predicted_test= svc_radial.predict(X_test)

cnf_matrix = confusion_matrix(y_train, predicted_train)
print("\nConfusion Matrix of SVM Training set : ")
print(cnf_matrix)

cnf_matrix = confusion_matrix(y_test, predicted_test)
print("\nConfusion Matrix of SVM Test set : ")
print(cnf_matrix)

print("\nAccuracy of SVM Train set : {0:.4f}".format(accuracy_score(y_train, predicted_train)))
print("\nAccuracy of SVM Test set : {0:.4f}".format(accuracy_score(y_test, predicted_test)))

print("\n*** Running KNN Model ****\n")
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
predicted_train = knn.predict(X_train)
predicted_test = knn.predict(X_test)

cnf_matrix = confusion_matrix(y_train, predicted_train)
print("\nConfusion Matrix of KNN Training set : ")
print(cnf_matrix)

cnf_matrix = confusion_matrix(y_test, predicted_test)
print("\nConfusion Matrix of KNN Test set : ")
print(cnf_matrix)

print("\nAccuracy of KNN Train set : {0:.4f}".format(accuracy_score(y_train, predicted_train)))
print("\nAccuracy of KNN Test set : {0:.4f}".format(accuracy_score(y_test, predicted_test)))

