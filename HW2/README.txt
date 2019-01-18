-----------------------------------------------------
=== Assignment 2 CS 6375.501: Machine Learning ===
-----------------------------------------------------
Name: Sharayu Mantri, Dhwani Raval
Email: ssm171330@utdallas.edu, dsr170230@utdallas.edu
NetID: ssm171330, dsr170230

Programming Language Used: Python

There are 4 text files and 2 csv files in the Assignment1 folder:
1. README.txt: 
	This file contains the information about team members and running instructions for the assignment.
	
2. RESULTS.txt: 
	This file contains the information about the technique used for a given dataset and its corresponding accuracy.
	
3. The datasets are present in /dataset:
	a. adult.names
	b. adultTest.data
	c. adultTrain.data
	d. bikeRentalHourlyTest.csv
	e. bikeRentalHourlyTrain.csv
	f. bikeSharingDataset.names.txt
	
There are 2 python files available in /sourceCode in Assignment1:
1. Q1_Census_Income.py: 
	Contains the code to train on adultTrain.data and to test on adultTest.data using NN, SVM Radial, KNN
	
2. Q2_Bike_Rental.py: 
	Contains the code to train on bikeRentalHourlyTrain.csv and to test on bikeRentalHourlyTest.csv using using neural networks, linear regression (you may use regularization techniques – lasso and ridge), and KNN.


-----------------------------------------------------
== USAGE: HOW TO RUN THE PROGRAM  ==
-----------------------------------------------------

Go to HW2 Directory

Q1.  Run 
python ./sourcecode/Q1_Census_Income.py './dataset/adultTrain.data' './dataset/adultTest.data'


Q2. Run 
python Q2_Bike_Rental.py
