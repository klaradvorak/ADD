import nltk,os, sys, glob, re, math, random

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

dirPath = 'C:\\Users\\klara\\PycharmProjects\\ADD\\venv\\Include\\sentiment labelled sentences\\'
os.chdir(dirPath)
myFiles = glob.glob('*.txt')

allDocumentsContent = []
with open(os.path.join(dirPath, 'amazon_cells_labelled.txt'), 'r') as reader:
    allDocumentsContent = allDocumentsContent + reader.readlines()

def textPreproccessing(text, setOfAllWords):
    lemmatizer = WordNetLemmatizer()
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
            setOfAllWords.add(token)
    return preproccessedTokens

def isFeatured(review):
    reviewWords = set(review)
    wordsOccured = {}
    for word in setOfAllWords:
        wordsOccured[word] = (word in reviewWords)
    return wordsOccured

random.shuffle(allDocumentsContent)
listOfAllContent = []
setOfAllWords = set()
for review in allDocumentsContent:
    preproccessedList = textPreproccessing(review, setOfAllWords)
    listOfAllContent.append((preproccessedList[:-1], preproccessedList[-1]))

# Train Naive Bayes classifier
featureSet = [(isFeatured(r), c) for (r,c) in listOfAllContent]
dataSet = train_test_split(featureSet, test_size= 0.2)
classifier = nltk.NaiveBayesClassifier.train(dataSet[0])

# Test the classifier
print(nltk.classify.accuracy(classifier, dataSet[1]))

# Show the most important features as interpreted by Naive Bayes
classifier.show_most_informative_features(10)
