#!/usr/bin/python
import numpy
import lda
import lda.datasets
import pickle 
import sys
from pandas import *


dfFile = open(sys.argv[1],"r")
lists_of_list = json.load(dfFile)
dfFile.close()

wordVectorFile = open(sys.argv[2],"r")
wordVector = json.load(wordVectorFile)
wordVectorFile.close()

docNamesFile = open(sys.argv[3], "r")
docs = json.load(docNamesFile)
docNamesFile.close()

titlesFile = open(sys.argv[4], "r")
titles = json.load(titlesFile)
titlesFile.close()

docFreqArray = numpy.array(lists_of_list)
print("shape: {}\n".format(docFreqArray.shape))
model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
model.fit(docFreqArray)

topic_word = model.topic_word_
print("type(topic_word): {}".format(type(topic_word)))
print("shape: {}".format(topic_word.shape))
##
for n in range(5):
    sum_pr = sum(topic_word[n,:])
    print("topic: {} sum: {}".format(n, sum_pr))

#top 5 words for each topic
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(wordVector)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_
print("type(doc_topic): {}".format(type(doc_topic)))
print("shape: {}".format(doc_topic.shape)) 

for n in range(5):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}\n{}...".format(n,topic_most_pr,titles[n][:50]))
