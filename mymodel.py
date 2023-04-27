import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import pickle
import warnings   # ignores pink warnings
warnings.filterwarnings('ignore')

phish_data = pd.read_csv('phishing_site_urls.csv')
# print("data loaded")
# phish_data.isnull().sum()
# print(phish_data.head(5))
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
# print('Getting words tokenized ...')
phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t))
# print(phish_data.head(5))
stemmer = SnowballStemmer("english")   # choose a language
# print('Getting words stemmed ...')
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
# print(phish_data.head(5))
# print('Getting joining words ...')
phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))
# print(phish_data.head(5))
cv = CountVectorizer()
feature = cv.fit_transform(phish_data.text_sent)
trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label, test_size=0.2)
lr = LogisticRegression()
lr.fit(trainX, trainY)
# print('Training Accuracy :',lr.score(trainX,trainY))
# print('Testing Accuracy :',lr.score(testX,testY))
# print(lr.score(testX,testY))

site = feature
pipeline_ls = make_pipeline(CountVectorizer(tokenizer=RegexpTokenizer(r'[A-Za-z]+').tokenize, stop_words='english'),
                            LogisticRegression())
trainU, testU, trainV, testV = train_test_split(phish_data.URL, phish_data.Label)
h = pipeline_ls.fit(trainU, trainV)
pipeline_ls.score(testU, testV)

pickle.dump(pipeline_ls, open('phishing_k.pkl', 'wb'))
loaded_model = pickle.load(open('phishing_k.pkl', 'rb'))
