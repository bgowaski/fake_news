import numpy as np
from pyspark import SparkContext
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
sc = SparkContext(‘local[4]’)
data = sc.textFile('project/abcnews-date-text.csv')
def get_sent(sentence):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(sentence)
out = data.map(lambda x: (x,get_sent(x)))
corp = out.filter(lambda x: x[1]["neg"]>0.75).map(lambda x: x[0]).flatMap(lambda x: x.split())

def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])
        
pairs = make_pairs(corp)

word_dict = {}

for word_1, word_2 in pairs:
    if word_1 in word_dict.keys():
        word_dict[word_1].append(word_2)
    else:
        word_dict[word_1] = [word_2]

first_word = np.random.choice(corp)
chain = [first_word]
n_words = 30

for i in range(n_words):
    chain.append(np.random.choice(word_dict[chain[-1]]))

' '.join(chain)
