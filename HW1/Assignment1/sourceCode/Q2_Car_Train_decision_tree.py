
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import os


# In[6]:


#Fetches the current directory
print(os.getcwd())


#Set the directory path to where the code is located in the system
#Change the current directory path to your path
#os.chdir('Mention your directory path here')
#print(os.getcwd())

le=LabelEncoder()
tle=LabelEncoder()

#Read training data in balance_data
balance_data = pd.read_csv('../dataset/carTrainData.csv', sep= ',')

#Read test data in balance_test_data
balance_test_data = pd.read_csv('../dataset/carTestData.csv', sep= ',')

print ("Dataset Lenght:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)
print(balance_data.columns.values.tolist())

    for column in balance_data:
    if balance_data[column].dtypes == 'object':
        data=balance_data[column]
        le.fit(data.values)
        balance_data[column] = le.transform(balance_data[column])
        
for column in balance_test_data:
    if balance_test_data[column].dtypes == 'object':
        testdata=balance_test_data[column]
        tle.fit(testdata.values)
        balance_test_data[column] = tle.transform(balance_test_data[column])


# In[4]:


X_train = balance_data.values[:, :6]
y_train = balance_data.values[:,6]

X_test = balance_test_data.values[:, :6]
y_test = balance_test_data.values[:,6]

#Using gini index in decision tree classifier
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                                  max_depth=4, min_samples_leaf=6)
clf_gini.fit(X_train, y_train)


#Using information gain in decision tree classifier
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                     max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

#Using random forest classifier
clf_rf = RandomForestClassifier()
trained_model = clf_rf .fit(X_train, y_train)


# In[5]:


import graphviz


# In[6]:


dot_data = tree.export_graphviz(clf_gini, out_file=None)
dot_data_entropy = tree.export_graphviz(clf_entropy, out_file=None)


# In[7]:


graph = graphviz.Source(dot_data)
graph_entropy = graphviz.Source(dot_data_entropy)


# In[8]:


feature_names =  ['buying price', 'maint cost', 'doors', 'persons', 'lug_boot', 'safety']
target_names = ['unacc', 'acc', 'good', 'vgood']
dot_data = tree.export_graphviz(clf_gini, out_file=None, 
                         feature_names=feature_names,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)

dot_data_entropy = tree.export_graphviz(clf_entropy, out_file=None, 
                         feature_names=feature_names,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)


# graph = graphviz.Source(dot_data)
# graph

# In[10]:


feature_names =  ['buying price', 'maint cost', 'doors', 'persons', 'lug_boot', 'safety']
target_names = ['unacc', 'acc', 'good', 'vgood']

#gini index test set predictions
y_pred = clf_gini.predict(X_test)

#Information gain index test set predictions
y_pred_entropy = clf_entropy.predict(X_test)

#Random forest classifier test set predictions
y_pred_rf = clf_rf.predict(X_test)


# In[24]:


clf_gini.predict_proba(X_test)
clf_entropy.predict_proba(X_test)
clf_rf.predict_proba(X_test)


# In[11]:


print ("Accuracy of gini index using decision tree classifier is ")
print(accuracy_score(y_test,y_pred)*100)
print( "Train Accuracy :: ", accuracy_score(y_train, clf_gini.predict(X_train)))
print( "Test Accuracy  :: ", accuracy_score(y_test, y_pred))
print( "Confusion matrix\n", confusion_matrix(y_test, y_pred))


# In[12]:


print ("Accuracy of information gain using decision tree classifier is ")
print(accuracy_score(y_test,y_pred_entropy)*100)
print( "Train Accuracy :: ", accuracy_score(y_train, clf_entropy.predict(X_train)))
print( "Test Accuracy  :: ", accuracy_score(y_test, y_pred_entropy))
print( "Confusion matrix\n", confusion_matrix(y_test, y_pred_entropy))


# In[13]:


print ("Accuracy of random forest is ")
print(accuracy_score(y_test,y_pred_rf)*100)
print( "Train Accuracy :: ", accuracy_score(y_train, clf_rf.predict(X_train)))
print( "Test Accuracy  :: ", accuracy_score(y_test, y_pred_rf))
print( " Confusion matrix\n", confusion_matrix(y_test, y_pred_rf))

