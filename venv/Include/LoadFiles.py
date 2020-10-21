import nltk,os, sys, glob, re, math

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

dirPath = 'C:\\Users\\klara\\PycharmProjects\\ADD\\venv\\Include\\sentiment labelled sentences\\'
os.chdir(dirPath)
myFiles = glob.glob('*.txt')

allDocumentsContent = []
with open(os.path.join(dirPath, 'amazon_cells_labelled.txt'), 'r') as reader:
    allDocumentsContent = allDocumentsContent + reader.readlines()
#print(allDocumentsContent)

#splits the data for 80% training set and 20% testing set
dataSet = train_test_split(allDocumentsContent, test_size= 0.2)
#calculate the total number of reviews
totalNofReviews = len(dataSet[0])

#normalization function
def textPreproccessing(text):
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

    return preproccessedTokens

invertedIndex = {}

#create inverted indext from the data
for line in dataSet[0]:
    #preprocessing
    tokens = textPreproccessing(line)
    #saves and removes positive/negative review value from tokens
    value = int(tokens[-1])
    tokens.pop()

    #stores tokens in dictionary with the possitive/negative value of its reviews
    for token in tokens:
        if invertedIndex.keys().__contains__(token):
            invertedIndex.get(token).append(value)
        else:
            invertedIndex[token] = [value]

print(invertedIndex)

def tfCalcualtion (review):
    tfMatrix = {}
    for token in review:
        if token in tfMatrix:
            tfMatrix[token] += 1
        else:
            tfMatrix[token] = 1
    
    return tfMatrix

def idfCalculation(freqMatrix, totalNumberOfReviews):
    idfMatrix = {}

    for token in freqMatrix.keys():
        idfMatrix[token] = math.log(totalNumberOfReviews / float(len(freqMatrix[token])))

    return idfMatrix

idfMatrix = idfCalculation(invertedIndex, totalNofReviews)

testTFIDF = {}
for review in dataSet[1]:
    listOfTokens = textPreproccessing(review)
    tfMatrix = tfCalcualtion(listOfTokens[:-1])
    for tfScore in tfMatrix:
        testTFIDF[tfScore] = (1 + math.log(tfMatrix[tfScore])) * idfMatrix[token]

print(testTFIDF)