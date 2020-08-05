from nltk import tokenize
from nltk import probability
from nltk import corpus
from nltk import stem
from matplotlib import pyplot as plt
import os
import nltk
from itertools import combinations


def doTheThing(fileContents):
    #    TOKENIZATION
    tokenizedWords = tokenize.word_tokenize(fileContents)

    #     STOPWORDS
    filteredWords=[]
    stop_words = set(corpus.stopwords.words('english'))
    for w in tokenizedWords:
        if w not in stop_words:
            filteredWords.append(w)

    #     FREQUENCY DISTRIBUTION
    freqDist = probability.FreqDist(tokenizedWords)

    #     STEMING
    ps = stem.PorterStemmer()
    stemmedWords = []
    for w in filteredWords:
        stemmedWords.append(ps.stem(w))

    #     LEMMATIZATION
    wnl = stem.WordNetLemmatizer()
    lemmatizedWords = []
    for w in filteredWords:
        lemmatizedWords.append(wnl.lemmatize(w, "v"))
    return [tokenizedWords,filteredWords,freqDist,stemmedWords,lemmatizedWords]

files = {}
for f in os.listdir('files'):
    with open('files/'+f, 'r') as computer:
        files[f] = doTheThing(computer.read())
        computer.close()



#COSINE SIMILARITY - VECTORS
def cosineSimilarity(X_set, Y_set):
    l1 = []
    l2 = []
    X_set = set(X_set)
    Y_set = set(Y_set)
    rvector = X_set.union(Y_set)
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
    
    #COSINE FORMULA  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5) 
    return cosine

filesKeys = []
for key in files:
    filesKeys.append(key)

for X, Y in combinations(filesKeys, 2):
    print("Cosine similarity of " + str(X) + " and " + str( Y ) + " ", cosineSimilarity(files[X][4], files[Y][4]))

#             PLOT
files['computer.txt'][2].plot(30, cumulative=False)