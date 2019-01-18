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

def getData(path, delimeter, testSplit=0.25):
    balance_data = pd.read_csv(path, sep= delimeter)
    le=LabelEncoder()
    for column in balance_data:
        if balance_data[column].dtypes == 'object':
            data=balance_data[column]
            le.fit(data.values)
            balance_data[column] = le.transform(balance_data[column])
    
    x_train = balance_data.values[:, 1:]
    y_train = balance_data.values[:,0]
    train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=float(testSplit))
    return train_x, test_x, train_y, test_y

def getCrossValidatedClassifiedData(name, train_x, test_x, train_y, test_y):
    clf = selectClassifier(name)
    print("Classifier selected : ", name)
    clf.fit(train_x,train_y)
    y_pred = clf.predict(test_x)
    conf_mat = confusion_matrix(test_y,y_pred)
    print("Confusion Matrix : ")
    print(conf_mat)
    print("Accuracy : ",accuracy_score(test_y,y_pred))
    print("\n")


def selectClassifier(name):
    if name == 'GradientBoosting':
        return GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.01)
    elif name == 'DecisionTreeGini':
        return DecisionTreeClassifier(criterion = "gini", random_state = 100,
                                      max_depth=4, min_samples_leaf=6)
    elif name == 'DecisionTreeEntropy':
        return DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                     max_depth=3, min_samples_leaf=5)
    elif name == 'RandomForest':
        return RandomForestClassifier()
    elif name == 'NeuralNetwork':
        return MLPClassifier(hidden_layer_sizes=(40,250,250,40), max_iter=1000)
    elif name == 'KNN':
        return KNeighborsClassifier(n_neighbors=10)
    else:
        return None

if(len(sys.argv) !=5 ):
    print("Command is python Q1.py __INPUT_DATA_SET_PATH__ __INPUT_SEPERATION_DELIMITER__  __INPUT_CLUSTERING_TECHNIQUE__ __TEST_SPLIT__")
    print("Example : python Q1.py './Assingment4/dataset/ThoraricSurgery.csv' ',' 'NeuralNetwork' , '0.25'")


#print('\nOutput for: python ' + sys.argv[0] + ' ' + sys.argv[1] + ' ' + sys.argv[2] + ' '+sys.argv[3] + ' ' + sys.argv[4] )

train_x, test_x, train_y, test_y = getData(sys.argv[1], sys.argv[2], sys.argv[4])
getCrossValidatedClassifiedData(sys.argv[3], train_x, test_x, train_y, test_y)

#getCrossValidatedClassifiedData('NeuralNetwork', balance_data, x_train, y_train)
#getCrossValidatedClassifiedData('GradientBoosting', balance_data, x_train, y_train)
#getCrossValidatedClassifiedData('DecisionTreeGini', balance_data, x_train, y_train)
#getCrossValidatedClassifiedData('DecisionTreeEntropy', balance_data, x_train, y_train)
#getCrossValidatedClassifiedData('RandomForest', balance_data, x_train, y_train)
#getCrossValidatedClassifiedData('KNN', balance_data, x_train, y_train)

