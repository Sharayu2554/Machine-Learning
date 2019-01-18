-----------------------------------------------------
=== Assignment 4 CS 6375.501: Machine Learning ===
-----------------------------------------------------
Name: Sharayu Mantri, Dhwani Raval
Email: ssm171330@utdallas.edu, dsr170230@utdallas.edu
NetID: ssm171330, dsr170230

Programming Language Used: Python

There are 2 text files in the Assignment 4 folder and 3 datasets in /dataset folder:

1. README.txt: 
	This file contains the information about team members and running instructions for the assignment.
	
2. RESULTS.txt: 
	This file contains the corresponding output.
	
3. The datasets are present in /dataset:
	a. ThoraricSurgery.csv
	b. Sales_Transactions_Dataset_Weekly.csv
	c. Concrete_Data.xls
	
There are 3 python files available in /sourcecode in Assignment 4:
1. Q1.py: 
	Contains the code to read and clean the dataset Thoracic Surgery Data and perform a simple cross validation using the classification method and print the confusion matrix.
	
2. Q2.py
	Contains the code to read and clean the dataset Sales_Transactions_Dataset_Weekly and using  K-Means clustering, create clusterings of size k=2, 10. For each clustering it prints TWSS/TSS ratio.

3. Q3.py
	Contains the code to read and clean the Concrete Compressive Strength Data Set and perform a simple cross validation using any method. It prints out the MSE.

-----------------------------------------------------
== USAGE: HOW TO RUN THE PROGRAM  ==
-----------------------------------------------------
Q1
python Q1.py '../dataset/ThoraricSurgery.csv' ',' 'NeuralNetwork' '0.25'
python Q1.py '../dataset/ThoraricSurgery.csv' ',' 'KNN' '0.25'
python Q1.py '../dataset/ThoraricSurgery.csv' ',' 'GradientBoosting' '0.25'
python Q1.py '../dataset/ThoraricSurgery.csv' ',' 'DecisionTreeGini' '0.25'
python Q1.py '../dataset/ThoraricSurgery.csv' ',' 'DecisionTreeEntropy' '0.25'
python Q1.py '../dataset/ThoraricSurgery.csv' ',' 'RandomForest' '0.25'

Q2
python Q2.py '../dataset/Sales_Transactions_Dataset_Weekly.csv' ',' 'K-Means'
python Q2.py '../dataset/Sales_Transactions_Dataset_Weekly.csv' ',' 'H-Clustering'
python Q2.py '../dataset/Sales_Transactions_Dataset_Weekly.csv' ',' 'GMM'

Q3
python Q3.py '../dataset/Concrete_Data.xls'
