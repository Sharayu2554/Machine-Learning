-----------------------------------------------------
=== Assignment 3 CS 6375.501: Machine Learning ===
-----------------------------------------------------
Name: Sharayu Mantri, Dhwani Raval
Email: ssm171330@utdallas.edu, dsr170230@utdallas.edu
NetID: ssm171330, dsr170230

Programming Language Used: Python

There are 2 text files in the Assignment3 folder and 2 datasets files in /dataset

1. README.txt: 
	This file contains the information about team members and running instructions for the assignment.
	
2. RESULTS.txt: 
	This file contains the corresponding output.
	
3. The datasets are present in /dataset:
	a. dataset_Facebook.csv
	b. Frogs_MFCCs.csv
	
There ia 1 python file available in /sourcecode
1. Clustering.py
	Contains the code to analyze the given two datasets using the three methods to determine the optimal number of clusters: 
		a. K-Means
		b. H-Clustering
		c. Gaussian Mixture Model	

-----------------------------------------------------
== USAGE: HOW TO RUN THE PROGRAM  ==
-----------------------------------------------------
Run the following for Anuran Calls dataset
1. python ./sourcecode/Clustering.py './dataset/Frogs_MFCCs.csv' ',' 'K-Means'
2. python ./sourcecode/Clustering.py './dataset/Frogs_MFCCs.csv' ',' 'H-Clustering'
3. python ./sourcecode/Clustering.py './dataset/Frogs_MFCCs.csv' ',' 'GMM'

Run the following for Facebook Post metrics dataset
1. python ./sourcecode/Clustering.py './dataset/dataset_Facebook.csv' ';' 'K-Means'
2. python ./sourcecode/Clustering.py './dataset/dataset_Facebook.csv' ';' 'H-Clustering'
3. python ./sourcecode/Clustering.py './dataset/dataset_Facebook.csv' ';' 'GMM'

