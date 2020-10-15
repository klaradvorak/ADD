import nltk,os, sys, glob, re, math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#normalization
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

dirPath = 'C:\\Users\\klara\\PycharmProjects\\ADD\\venv\\Include\\sentiment labelled sentences\\'
os.chdir(dirPath)
myFiles = glob.glob('*.txt')

allDocumentsContent = []
with open(os.path.join(dirPath, 'amazon_cells_labelled.txt'), 'r') as reader:
    allDocumentsContent = allDocumentsContent + reader.readlines()

#print(allDocumentsContent)

invertedIndex = {}
totalNOfPosivitveReviewes = 0
totalNOfNegativeReviews = 0
totalNofPositiveTerms = 0
totalNofNegativeTerms = 0
#create inverted indext from the data
for line in allDocumentsContent:
    #preprocessing
    tokens = textPreproccessing(line)
    #saves and removes positive/negative review value from tokens
    value = int(tokens[-1])
    tokens.pop()

    #saves the total number of positive and negative reviews
    if value == 0:
        totalNOfNegativeReviews += 1
    elif value == 1:
        totalNOfPosivitveReviewes += 1
    else:
        print("Error unexpected value in tokenization")

    #stores tokens in dictionary with the possitive/negative value of its reviews
    for token in tokens:
        if invertedIndex.keys().__contains__(token):
            invertedIndex.get(token).append(value)
        else:
            invertedIndex[token] = [value]

        if value == 0:
            totalNofNegativeTerms += 1
        elif value == 1:
            totalNofPositiveTerms += 1
        else:
            print("Error unexpected value")


print(invertedIndex)
#print(totalNOfNegativeReviews)

"""all code above has to be executed only once for the set, thinker with storing result in JSON or something"""
#tf.idf weighting

tfIDFweight = {}
totalNumberOfReviews = totalNOfNegativeReviews + totalNOfPosivitveReviewes

print(totalNofPositiveTerms, totalNofNegativeTerms)

for term in invertedIndex:
    negative = 0
    positive = 0
    for value in invertedIndex.get(term):
        if value == 0:
            negative += 1
        else:
            positive += 1

    tfPositive = positive / totalNofPositiveTerms
    tfNegative = negative / totalNofNegativeTerms

    idf = math.log10(totalNumberOfReviews / len(invertedIndex.get(term)))

    tfIDFweight[term] = [(tfPositive * idf), (tfNegative * idf)]

print(tfIDFweight)