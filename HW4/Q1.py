
# coding: utf-8

# In[3]:

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

def getData(path, delimeter):
    balance_data = pd.read_csv(path, sep= delimeter)
    le=LabelEncoder()
    for column in balance_data:
        if balance_data[column].dtypes == 'object':
            data=balance_data[column]
            le.fit(data.values)
            balance_data[column] = le.transform(balance_data[column])
    
    x_train = balance_data.values[:, 1:]
    y_train = balance_data.values[:,0]
    train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.25)
    return balance_data, x_train, y_train

def getCrossValidatedClassifiedData(name, balance_data, x_train, y_train):
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state=0)
    clf = selectClassifier(name)
    scores = cross_val_score(clf, x_train, y_train, cv=cv)
    y_pred = cross_val_predict(clf,x_train,y_train,cv=4)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(cv)
    conf_mat = confusion_matrix(y_train,y_pred)
    print(conf_mat)


def selectClassifier(name):
    if name is 'GradientBoosting':
        return GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.01)
    elif name is 'DecisionTreeGini':
        return DecisionTreeClassifier(criterion = "gini", random_state = 100,
                                      max_depth=4, min_samples_leaf=6)
    elif name is 'DecisionTreeEntropy':
        return DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                     max_depth=3, min_samples_leaf=5)
    elif name is 'RandomForest':
        return RandomForestClassifier()
    elif name is 'NeuralNetwork':
        return MLPClassifier(hidden_layer_sizes=(40,250,250,40), max_iter=1000)
    elif name is 'KNN':
        return KNeighborsClassifier(n_neighbors=10)
    else:
        return None

if(len(sys.argv) !=4 ):
    print("Command is Python Q1.py __INPUT_DATA_SET_PATH__ __INPUT_SEPERATION_DELIMITER__  __INPUT_CLUSTERING_TECHNIQUE__")
    print("Example : Python Q1.py './Assingment4/dataset/ThoraricSurgery.csv' ',' 'NeuralNetwork'")


print('\nOutput for: python ' + sys.argv[0] + ' ' + sys.argv[1] + ' ' + sys.argv[2] + ' '+sys.argv[3])

balance_data, x_train, y_train = getData(sys.argv[1], sys.argv[2])
getCrossValidatedClassifiedData(sys.argv[3], balance_data, x_train, y_train)

#getCrossValidatedClassifiedData('NeuralNetwork', balance_data, x_train, y_train)
#getCrossValidatedClassifiedData('GradientBoosting', balance_data, x_train, y_train)
#getCrossValidatedClassifiedData('DecisionTreeGini', balance_data, x_train, y_train)
#getCrossValidatedClassifiedData('DecisionTreeEntropy', balance_data, x_train, y_train)
#getCrossValidatedClassifiedData('RandomForest', balance_data, x_train, y_train)
#getCrossValidatedClassifiedData('KNN', balance_data, x_train, y_train)

