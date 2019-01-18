
# coding: utf-8

# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
import os 
import re


# In[17]:


#Fetches the current directory
print(os.getcwd())


#Set the directory path to where the code is located in the system
#Change the current directory path to your path
#os.chdir('Mention your directory path here')
#print(os.getcwd())

# Read the training data from the Assignment1/dataset folder in the dataset
dataset = pd.read_csv('../dataset/textTrainData.txt', sep='\t', encoding='latin1')

#Then we read the test data from the Assignment1/dataset folder in the testset
testset = pd.read_csv('../dataset/textTestData.txt', sep='\t', encoding='latin1')

print(dataset.iloc[0:10])
dataset.count()
testset.count()


# In[18]:


X_train = dataset.values[:, :1]
y_train = dataset.values[:,1]

X_test = testset.values[:, :1]
y_test = testset.values[:, 1]


# In[19]:


# A function that will split the text based upon sentiment
def get_text(sentences, sentiments, score):
  # Join together the text in the reviews for a particular sentiment
  # We lowercase to avoid "Not" and "not" being seen as different words, for example
   
    s = ""
    for index in range(len(sentences) -1):
        if score == -1:
            s = s + sentences[index][0].lower()
        elif (sentiments[index] == score):
            s = s + sentences[index][0].lower()
        else:
            s = s 
    return s

X_train_data = get_text(X_train, y_train, -1)
X_test_data = get_text(X_train, y_train, -1)

X_train_text_1 = get_text(X_train, y_train, 1)
X_test_text_1 = get_text(X_test, y_test, 1)

X_train_text_0 = get_text(X_train, y_train, 0)
X_test_text_0 = get_text(X_test, y_test, 0)


# In[20]:


# A function that will count word frequency for each sample
def count_text(text):
  # Split text into words based on whitespace.  Simple but effective
  words = re.split("\s+", text)
  # Count up the occurence of each word
  return Counter(words)

# Generate the word counts for each sentiment
negative_counts = count_text(X_train_text_0)

# Generate word counts for positive tone
positive_counts = count_text(X_train_text_1)


# In[21]:


# A function to calculate a count of a given classification
def get_y_count(sentiments,score):
  # Compute the count of each classification occuring in the data
  # return len([r for r in reviews if r[1] == str(score)])
    c = 0
    for index in range(len(sentiments)):
        if sentiments[index] == score:
            c = c + 1
    
    return c

positive_sentence_count = get_y_count(y_train, 1)
negative_sentence_count = get_y_count(y_train, 0)


# In[22]:


# These are the class probabilities
prob_positive = positive_sentence_count / len(y_train)
prob_negative = negative_sentence_count / len(y_train)
prob_positive


# In[23]:


# A function that will, given a text example, allow us to calculate the probability
# of a positive or negative review

def make_class_prediction(text, counts, class_prob, class_count):
  prediction = 1
  text_counts = Counter(re.split("\s+", text))
  for word in text_counts:
      # Smooth the denominator counts to keep things even
      prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
  # Multiply by the probability of the class existing in the documents.
  return prediction * class_prob

print("Negative prediction: {0}".format(make_class_prediction(X_train[0][0], negative_counts, prob_negative, negative_sentence_count)))
print("Positive prediction: {0}".format(make_class_prediction(X_train[0][0], positive_counts, prob_positive, positive_sentence_count)))


# In[24]:


# A function that will actually make the prediction
def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities
    negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_sentence_count)
    positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_sentence_count)

    # A classification based on which probability is greater
    if negative_prediction > positive_prediction:
      return 0
    return 1

#print(make_decision(X_train[1][0], make_class_prediction))
#print(y_train[1])


# In[25]:


# Make predictions on the test data
predictions_train = []
for index in range(len(X_test)-1):
    predictions_train.append(make_decision(X_train[index][0], make_class_prediction))

predictions = []
for index in range(len(X_test)-1):
    predictions.append(make_decision(X_test[index][0], make_class_prediction))

# In[27]:


# Check the accuracy
accuracy = sum(1 for i in range(len(predictions_train)) if predictions_train[i] == y_train[i]) / float(len(predictions))
print("Accuracy of Naive Bayes on Training set : {0:.4f}".format(accuracy))


accuracy = sum(1 for i in range(len(predictions)) if predictions[i] == y_test[i]) / float(len(predictions))
print("Accuracy of Naive Bayes on Test set : {0:.4f}".format(accuracy))

