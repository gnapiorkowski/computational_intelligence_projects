import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

data=pd.read_csv('train.tsv', sep='\t')
def showDataPlot():
    Sentiment_count=data.groupby('Sentiment').count()
    plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
    plt.xlabel('Review Sentiments')
    plt.ylabel('Number of Review')
    plt.show()
# showDataPlot()

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Phrase'])

x_train, x_test, y_train, y_test = train_test_split(
    text_counts, data['Sentiment'], test_size=0.3, random_state=1)

clf = MultinomialNB().fit(x_train, y_train)
predicted= clf.predict(x_test)
print("MultinomialNB acc (no TF-IDF):",metrics.accuracy_score(y_test, predicted))

tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['Phrase'])

x_train, x_test, y_train, y_test = train_test_split(
    text_tf, data['Sentiment'], test_size=0.3, random_state=123)

clf = MultinomialNB().fit(x_train, y_train)
predicted= clf.predict(x_test)

print("MultinomialNB acc (TF-IDF):",metrics.accuracy_score(y_test, predicted))