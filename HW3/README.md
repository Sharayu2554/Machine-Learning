We have seen three methods for performing clustering: K-Means, H-Clustering, and Gaussian Mixture Models. Each of these methods will take a dataset and, given a number of clusters to look for,cluster the datapoints into these clusters; they all provide a way of obtaining labels (cluster assignments) for the datapoints.

Your goal in this assignment is to analyze two datasets (see below) using the three methods, in an effort to determine the optimal number of clusters. To do this, you need a way of measuring how well a clustering works for a given number of clusters. One common approach is to compare the total within sum of squares to the total sum of squares of the data (see attached formula sheet).  We want this to be small; you should notice that this ratio decreases as the number of clusters increases (so the smallest value should occur for the largest number of clusters).

What to submit: You will submit a Python script which will do the following for each of the two datasets:

Read and clean (if necessary) the dataset – please be sure your script reads the dataset from the directory in which it is being run so the TA can run your script easily.
Calculate the total sum of squares for the dataset
For each of the three clustering methods:
For k=1, …, 10:
Perform a clustering
Using the clustering, calculate the total within sum of squares
Print out the ratio of total within sum of squares/total sum of squares
 

Please submit your script by midnight, April 1.

Datasets to use:

Anuran Calls (http://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29) – use columns 2-22 for the clustering

Facebook Post metrics (http://archive.ics.uci.edu/ml/datasets/Facebook+metrics)
