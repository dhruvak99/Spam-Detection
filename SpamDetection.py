import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

#import the spam dataset
data = pd.read_csv('dataset',sep='\t',names=['label','message'])
#we will print the head of the dataset
print(data.head(10))
#we will add a new column to the dataset called 'length' which is the length of the text message
data['length'] = data['message'].apply(len)
#we will plot a graph which will have two coloumns
#first column is a graph for the HAM messages
#second column is a graph for the SPAM messages , both are differentiated based on their lengths
data.hist(column='length',by='label',bins=50,figsize=(12,6))
plt.show()
#we can notice the mean lengths of HAM and SPAM are different

#this will be our analyzer function for the CountVectorizer
def text_process(mess):
    """
    1.remove the punctuations
    2.join all the characters
    3.remove the stopwords
    """
    no_punctuations = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(no_punctuations)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
#split the dataset into training and testing data
mess_train,mess_test,label_train,label_test = train_test_split(data['message'],data['label'],test_size=0.2)
#we are using pipeline as it will carry out all the steps required
pipeline = Pipeline([
    ('bow',CountVectorizer(text_process))
    ,('tdidf',TfidfTransformer()),
    ('classifier',RandomForestClassifier(n_estimators=500,verbose=3))
])
pipeline.fit(mess_train,label_train)
predictions = pipeline.predict(mess_test)
#we will print the classification report of our model to find the accuracy
print(classification_report(label_test,predictions))
