-----------------------------------------------------
=== Assignment 1 CS 6375.501: Machine Learning ===
-----------------------------------------------------
Name: Sharayu Mantri, Dhwani Raval
Email: ssm171330@utdallas.edu, dsr170230@utdallas.edu
NetID: ssm171330, dsr170230

Programming Language Used: Python


Pre-Requisites :
Following libraries required to be installed on CS_GRADE_MACHINE

pip install --user  -U scikit-learn
pip install --user  -U numpy
pip install --user  -U spicy
pip install --user  -U graphviz


There are 4 text files and 2 csv files in the Assignment1 folder:
1. README.txt: 
	This file contains the information about team members and running instructions for the assignment.
	
2. RESULTS.txt: 
	This file contains the information about the technique used for a given dataset and its corresponding accuracy.
	
3. The datasets are present in /dataset:
	a. textTrainData.txt
	b. textTestData.txt
	c. carTrainData.csv
	d. carTestData.csv

	
There are 2 python files available in /sourceCode in Assignment1:
1. Q1_Text_Train_Naive_Bayes.py: 
	Contains the code to train textTrainData.txt using Naive Bayes
	
2. Q2_Car_Train_decision_tree.py: 
	Contains the code to train carTrainData.csv using Decision Trees - gini index, information gain and random forest

	
-----------------------------------------------------
== MANUAL CHANGES TO BE MADE BEFORE RUNNING THE CODE ==
-----------------------------------------------------
1. If the downloaded code is not in the current directory, then change the path by performing the following steps:
	a. Q1_Text_Train_Naive_Bayes.py
		Uncomment the following code and pass the directory path as its parameter.	
		#Set the directory path to where the code is located in the system
		#Change the current directory path to your path
		#os.chdir('Mention your directory path here')
		
	b. Q2_Car_Train_decision_tree.py
		Uncomment the following code and pass the directory path as its parameter.
		#Set the directory path to where the code is located in the system
		#Change the current directory path to your path
		#os.chdir('Mention your directory path here')

		
2. Overwrite the carTestData.csv and textTestData.txt files in /dataset in Assignment1



-----------------------------------------------------
== USAGE: HOW TO RUN THE PROGRAM  ==
-----------------------------------------------------
(NOTE: PLEASE ENSURE THAT THE LOCATION OF THE DATASET IS ACCESSIBLE TO THE PROGRAM.)

To run the program, Code is availiable in ./Assignment1/sourceCode ,in the terminal navigate to ./Assignment1/sourceCode where both the python files are available.

Replace your test Data sets with files in ./Assignment1/dataset/ with your files with same names as textTestData.txt, carTestData.csv

Q1.  Run 
python Q1_Text_Train_Naive_Bayes.py


Q2. Run 
python Q2_Car_Train_decision_tree.py



Run the program
-->
