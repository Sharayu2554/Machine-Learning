import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))

if(len(sys.argv) != 2):
	print("Command is python Q3.py __INPUT_TRAIN_FILE_PATH__ ")
	print("Example : python Q3.py '../dataset/Concrete_Data.xls' ")

train_Data = pd.read_excel(sys.argv[1])

headers=['cement','blast', 'flyAsh', 'water', 'superplasticiz', 'coarseAggregate', 'fineAggregate',
       'age', 'ccs']
train_Data.columns=headers
trainData_X=train_Data.drop(train_Data.columns[-1], axis=1)
trainData_Y=train_Data[train_Data.columns[-1]]
train_x, test_x, train_y, test_y=train_test_split(trainData_X,trainData_Y, test_size=0.25 )

data_X=train_x.append(test_x)
data_Y=train_y.append(test_y)

print('\nKNN')
KNNRegr = KNeighborsRegressor(n_neighbors=5)
KNNRegr.fit(train_x,train_y)
predictions = KNNRegr.predict(test_x)
MSE = sum((predictions - test_y)**2)/len(predictions)
print("Means square error using Kmeans with 5 neighbors : ",MSE)

kFold= KFold(n_splits=5)
kFold.get_n_splits(data_X)
MSE = 0
for train_index, test_index in kFold.split(data_X):
    KNNRegr = KNeighborsRegressor(n_neighbors=5)
    KNNRegr.fit(data_X.iloc[train_index], data_Y.iloc[train_index])
    predictions = KNNRegr.predict(data_X.iloc[test_index])
    MSE += sum((predictions - data_Y.iloc[test_index])**2)/len(predictions)
    
print("Means square error for 5 fold : ",MSE/5.0)
