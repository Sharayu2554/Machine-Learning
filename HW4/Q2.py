
# coding: utf-8

# In[ ]:


import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture

def algo(algo, numpyMatrix, TSS):
    result = 0;
    if algo is 'kmeans':
        result = kmeansClustering(numpyMatrix, TSS)
    elif algo is 'hclustering':
        result = hClustering(numpyMatrix, TSS)
    elif algo is 'guassian':
        result =  guassianMixtureModelClustering(numpyMatrix, TSS)
    return result

def guassianMixtureModelClustering(numpyMatrix, TSS):
    for cluster in range(1,11):
        clusteredMatrix =  GMMClustering(cluster, numpyMatrix)
        clusteredMapResponse = groupResponse(clusteredMatrix, numpyMatrix)
        clusteredMapMean = clusteredMean(clusteredMapResponse)
        TWSS = getTWSS(clusteredMapResponse, clusteredMapMean)
        print("K :" + str(cluster))
        #print("TWSS : " + str(TWSS))
        #print("TSS : " + str(TSS))
        print("result : " + str(round(TWSS,2)/round(TSS,2)) )
    return str(round(TWSS,2)/round(TSS,2))

def hClustering(numpyMatrix, TSS):
    Z = linkage(numpyMatrix, 'ward')
    for cluster in range(1,11):
        clusteredMatrix = HClustering(cluster, Z)
        clusteredMapResponse = groupResponse(clusteredMatrix, numpyMatrix)
        clusteredMapMean = clusteredMean(clusteredMapResponse)
        TWSS = getTWSS(clusteredMapResponse, clusteredMapMean)
        print("K :" + str(cluster))
        #print("TWSS : " + str(TWSS))
        #print("TSS : " + str(TSS))
        print("result : " + str(round(TWSS,2)/round(TSS,2)) )
    return str(round(TWSS,2)/round(TSS,2))

def kmeansClustering(numpyMatrix, TSS):
    for cluster in range(1,11):
        clusteredMatrix =  KMeansClustering(cluster, numpyMatrix)
        clusteredMapResponse = groupResponse(clusteredMatrix, numpyMatrix)
        clusteredMapMean = clusteredMean(clusteredMapResponse)
        TWSS = getTWSS(clusteredMapResponse, clusteredMapMean)
        print("K :" + str(cluster))
        #print("TWSS : " + str(TWSS))
        #print("TSS : " + str(TSS))
        print("result : " + str(round(TWSS,2)/round(TSS,2)) )
    return str(round(TWSS,2)/round(TSS,2))

def getTWSS(clusteredMapResponse, clusteredMapMean):
    TWSS = 0
    for key in clusteredMapResponse:
        #print("key :" + str(key))
        twsCluster = 0
        response = clusteredMapResponse.get(key)
        mean = clusteredMapMean.get(key)
        for row in response:
            for index in range(len(row)):
                twsCluster = twsCluster + (row[index] - mean[index]) * (row[index] - mean[index])
                #print("row[index] " + str(row[index])  + "mean[index] " + str(mean[index]))
        TWSS = TWSS + twsCluster
    return TWSS

def clusteredMean(clusteredMap):
    hashMap = {}
    for key in clusteredMap.keys():
        values = clusteredMap.get(key) 
        meanMatrix = np.mean(values, axis=0)
        hashMap[key] = meanMatrix
    return hashMap


def groupResponse(clusteredMatrix, originalMatrix):
    hashMap = {}
    resultMap =  {}
    i = 0
    for val in clusteredMatrix:
        result = originalMatrix[i]
        if val in hashMap:
            hashMap.get(val).append(i)
            resultMap.get(val).append(result)
        else:
            hashMap[val] = [i]
            resultMap[val] = [result]
        i = i + 1
    return resultMap


def totalSumSquareRawData(balance_data, meanMatrix):
    TSS = 0
    for index, row in balance_data.iterrows():
        columnIndex = 0
        for column in balance_data:
            TSS = TSS + (row[column] - meanMatrix[columnIndex] )* (row[column] - meanMatrix[columnIndex] )
            columnIndex = columnIndex + 1
    return TSS


def createColumnMatrix(column):
    matrix= {}
    for i in range(1,len(column)):
        matrix[i] = column[i]
    matrix = [(k,v) for k,v in matrix.items()]
    return matrix


def KMeansClustering(clusterNumber, data):
    kmeans = KMeans(n_clusters=clusterNumber)
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)
    return y_kmeans


def HClustering(clusterNumber, data):
    return fcluster(data, clusterNumber, criterion='maxclust')


def GMMClustering(clusterNumber, data):
    gmm = GaussianMixture(n_components=clusterNumber, covariance_type='tied')
    gmm.fit(data)
    y_gmm=gmm.predict(data)
    return  y_gmm


def getTSS(numpyMatrix):
    meanMatrix = np.mean(numpyMatrix, axis=0)
    TSS = totalSumSquareRawData(balance_data, meanMatrix)
    return TSS


def cleanData(balance_data):
    balance_data = balance_data.fillna(0)
    balance_data = balance_data.replace(np.inf, 0)
    np.where(balance_data.values >= np.finfo(np.float64).max)
    return balance_data


def encodeDataSet(data):
    le=LabelEncoder()
    for column in data:
        if data[column].dtypes == 'object':
            data_new=data[column]
            le.fit(data_new.values)
            data[column] = le.transform(data[column])
    return data


if(len(sys.argv) !=4 ):
    print("Command is Python Clustering.py __INPUT_DATA_SET_PATH__ __INPUT_SEPERATION_DELIMITER__  __INPUT_CLUSTERING_TECHNIQUE__")
    print("Example : Python Clustering.py '../dataset/Frogs_MFCCs.csv' ';' 'kmeans'")


print('\nOutput for: python ' + sys.argv[0] + ' ' + sys.argv[1] + ' ' + sys.argv[2] + ' '+sys.argv[3])
#print('\nDataset Path: ' + sys.argv[1])
#print('Dataset Deliimter: '+sys.argv[2])

balance_data = pd.read_csv(sys.argv[1], sep=sys.argv[2])
#balance_data = pd.read_csv('./dataset/Frogs_MFCCs.csv', sep=',')
#balance_data = pd.read_csv('./dataset/dataset_Facebook.csv', sep=';')

if(sys.argv[1]=="./dataset/Frogs_MFCCs.csv"):
    balance_data = balance_data.drop(['MFCCs_ 1', 'Family', 'Genus', 'Species', 'RecordID'], axis=1)
balance_data = encodeDataSet(balance_data)
balance_data = cleanData(balance_data)
numpyMatrix = balance_data.as_matrix()

print('\n*** Clustering used is ' + sys.argv[3] + ' ***\n')
if(sys.argv[3]=='K-Means'):
    clustering='kmeans'
elif (sys.argv[3]=='H-Clustering'):
     clustering='hclustering'
elif (sys.argv[3]=='GMM'):
     clustering='guassian'
else:
     print('Invalid Clustering Technique')


TSS = getTSS(numpyMatrix)
print("TSS:" + str(TSS)+"\n")


result = algo(clustering, numpyMatrix, TSS)

#print('Clustering Used: H-Clustering')
#result = algo('hclustering', numpyMatrix, TSS)


#print('Clustering Used: Gaussian Mixture Models')
#result = algo('guassian', numpyMatrix, TSS)


#TSS = getTSS(numpyMatrix)
#print(TSS)


#print('Clustering Used:  K-Means')
#result = algo('kmeans', numpyMatrix, TSS)


#print('Clustering Used: H-Clustering')
#result = algo('hclustering', numpyMatrix, TSS)


#print('Clustering Used: Gaussian Mixture Models')
#result = algo('guassian', numpyMatrix, TSS)


