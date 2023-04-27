import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import warnings # ignores pink warnings
warnings.filterwarnings('ignore')

phish_data = pd.read_csv('phishing_site_urls.csv')
print("data loaded")
#phish_data.isnull().sum()
#print(phish_data.head(5))
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
print('Getting words tokenized ...')
phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t))
#print(phish_data.head(5))
stemmer = SnowballStemmer("english") # choose a language
print('Getting words stemmed ...')
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
#print(phish_data.head(5))
print('Getting joining words ...')
phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))
#print(phish_data.head(5))
cv = CountVectorizer()
feature = cv.fit_transform(phish_data.text_sent)
trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label, test_size=0.20)
lr = LogisticRegression()
lr.fit(trainX,trainY)
threshold = 0.5
print(lr.predict(testX))
Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)
print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY), columns = ['Predicted:Bad', 'Predicted:Good'],index = ['Actual:Bad', 'Actual:Good'])
print('\nLogistic Regression -CLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY, target_names =['Bad','Good']))
print("multi")
mnb = MultinomialNB()
mnb.fit(trainX,trainY)
Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)
print('Training Accuracy :',mnb.score(trainX,trainY))
print('Testing Accuracy :',mnb.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(mnb.predict(testX), testY),columns = ['Predicted:Bad', 'Predicted:Good'],index = ['Actual:Bad', 'Actual:Good'])
print('\nMultinomialNB-CLASSIFICATION REPORT\n')
print(classification_report(mnb.predict(testX), testY,target_names =['Bad','Good']))


