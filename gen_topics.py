#!/usr/bin/python
import sys
import os
import json
import operator
import numpy
import nltk
from nltk.corpus import stopwords
import pickle

cachedStopWords = stopwords.words("english")
docFrequency = dict()

def main(argv):
    topicDict  = dict()
    docTitles = dict()
    for file in os.listdir(argv[1]):
        f = open(os.path.join(argv[1],file),"r")
        jsonDict = json.load(f)
        keywords = jsonDict[0]['keywords']
        descList = nltk.word_tokenize(jsonDict[0]['description'])
        descStopWordsRemoved = [word for word in descList if word not in cachedStopWords]
        docFrequency[file]=dict()
        docTitles[file]= ' '.join(sorted(jsonDict[0]['keywords']))
        for word in descStopWordsRemoved: 
            if word not in docFrequency[file]:
                docFrequency[file][word] = 1
            else:
                docFrequency[file][word]=docFrequency[file][word] + 1
    wordsVector=[]
 
    #prepare vector of words 
    for doc,words in sorted(docFrequency.items()):
        for word in words:
            if word not in wordsVector:
                wordsVector.append(word)
    list_of_lists=[]

    #prepare lists of lists of df for lda
    docNames = []
    for doc,words in sorted(docFrequency.items()):
        docNames.append(doc) 
        wordList=[]
        for word in wordsVector:
            try:
                wordList.append(words[word])
            except KeyError:
                wordList.append(0)
        list_of_lists.append(wordList)
 
    #dump doc freq matrix
    docFreqFile = open("docFrequency","w")
    json.dump(list_of_lists,docFreqFile)
    docFreqFile.close()

    #dump the wordvector
    wordVectorFile = open("wordVector", "w")
    json.dump(wordsVector,wordVectorFile)
    wordVectorFile.close()

    #dump doc name in sorted order
    docNamesFile = open("docNames","w")
    json.dump(docNames,docNamesFile)
    docNamesFile.close()

    #dump doc titles in sorted order
    docTitleFile = open("titles", "w")
    json.dump([docTitles[doc] for doc in sorted(docTitles)],docTitleFile)
    docTitleFile.close()

if __name__=="__main__":
    main(sys.argv)
