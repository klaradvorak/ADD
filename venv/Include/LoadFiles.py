import nltk,os, sys, glob, re, math, random
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

dirPath = 'C:\\Users\\klara\\PycharmProjects\\ADD\\venv\\Include\\sentiment labelled sentences\\'
os.chdir(dirPath)
myFiles = glob.glob('*.txt')

allDocumentsContent = []
for textFileName in myFiles:
    with open(os.path.join(dirPath, textFileName), 'r') as reader:
        allDocumentsContent = allDocumentsContent + reader.readlines()

setOfAllWords = set()

#preproccessing
def textPreproccessing(text, setOfAllWords):
    #initializes Lemmatizer
    lemmatizer = WordNetLemmatizer()
    #set of stop words based on Stop wrods library
    stopWords = set(stopwords.words('english'))
    preproccessedTokens = []

    #all characters to lower letter char
    text = text.lower()
    #remove punctuation
    text = re.sub(r'[^\w\s]', "", text)
    #tokenize : split in to individual tokens and stores in list
    tokens = nltk.word_tokenize(text)

    for token in tokens:
        #reduces word to a root synonym
        lemmetizeToken = lemmatizer.lemmatize(token)
        #removes stop words
        if not token in stopWords:
            preproccessedTokens.append(token)
            #adds values to set of all words in all documents
            setOfAllWords.add(token)
    return preproccessedTokens

#determine whether word if present from list of all words or not
def isFeatured(review):
    reviewWords = set(review)
    wordsOccured = {}
    for word in setOfAllWords:
        wordsOccured[word] = (word in reviewWords)
    return wordsOccured

def trainClassifier():
    #shoufless the position of reviews
    random.shuffle(allDocumentsContent)

    #initials variables
    listOfAllContent = []

    #for each review in all documents
    for review in allDocumentsContent:
        #applies preprocessing
        listOfTokens = textPreproccessing(review, setOfAllWords)
        listOfAllContent.append((listOfTokens[:-1], listOfTokens[-1]))

    # Train Naive Bayes classifier
    featureSet = [(isFeatured(r), c) for (r,c) in listOfAllContent]
    dataSet = train_test_split(featureSet, test_size= 0.2)
    classifier = nltk.NaiveBayesClassifier.train(dataSet[0])

    # Test the classifier
    return nltk.classify.accuracy(classifier, dataSet[1])

    # Show the most important features as interpreted by Naive Bayes
    #classifier.show_most_informative_features(10)

#testing
print("running")
accuracy = []
avg = 0
for number in range(10):
    numbber =trainClassifier()
    accuracy.append(numbber)
    avg = avg + numbber
print(accuracy)
print(avg/10)









