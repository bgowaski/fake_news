import numpy as np
from pyspark import SparkContext
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from time import time
import random

analyzer = SentimentIntensityAnalyzer()
sc = SparkContext('local[20]')

start = time()

myrdd = sc.textFile('dataset/abcnews-date-text.csv')
data = myrdd.repartition(20)
print("Num workers: ", 20)
print("Num partitions: ", data.getNumPartitions())

def get_sent(sentence):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(sentence)
    
out = data.map(lambda x: (x,get_sent(x)))
corp = out.filter(lambda x: x[1]["neg"]>0.5).map(lambda x: x[0]).flatMap(lambda x: x.split())
corpus = corp.take(1000000)
random.shuffle(corpus)
train_data = corpus[:int(len(corpus)*0.8)]
test_data = corpus[int(len(corpus)*0.8):]

end = time()
print("Data processing time:", (end-start))

def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])
        
pairs_train = make_pairs(train_data)
pairs_test = make_pairs(test_data)

word_dict_test = {}
word_dict_train = {}

for word_1, word_2 in pairs_train:
    if word_1 in word_dict_train.keys():
        word_dict_train[word_1].append(word_2)
    else:
        word_dict_train[word_1] = [word_2]

for word_1, word_2 in pairs_test:
    if word_1 in word_dict_test.keys():
        word_dict_test[word_1].append(word_2)
    else:
        word_dict_test[word_1] = [word_2]

count_words = 0
    
for i in word_dict_test.keys():
    try:
        for j in word_dict_test[i]:
            if j in word_dict_train[i]:
                count_words += 1
    except KeyError as e:
        continue

count_total =  len(word_dict_test.values())
cross_validation_score = count_words/count_total

print("Cross-validation score:", cross_validation_score*100, "%")  
