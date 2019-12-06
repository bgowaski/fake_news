import numpy as np
from pyspark import SparkContext
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from time import time

analyzer = SentimentIntensityAnalyzer()
sc = SparkContext('local[400]')

start = time()

myrdd = sc.textFile('dataset/small3.csv')
data = myrdd.repartition(400)
print("Num partitions: ", data.getNumPartitions())

def get_sent(sentence):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(sentence)
    
out = data.map(lambda x: (x,get_sent(x)))
corp = out.filter(lambda x: x[1]["neg"]>0.5).map(lambda x: x[0]).flatMap(lambda x: x.split())
corpus = corp.take(10000)

print("Execution time:", (time()-start))

def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])
        
pairs = make_pairs(corpus)

word_dict = {}

for word_1, word_2 in pairs:
    if word_1 in word_dict.keys():
        word_dict[word_1].append(word_2)
    else:
        word_dict[word_1] = [word_2]

neg_list = list()

for k in range(100):
    first_word = np.random.choice(corpus)
    chain = [first_word]
    n_words = 10

    for i in range(n_words):
        try:
            chain.append(np.random.choice(word_dict[chain[-1]]))
        except KeyError as e:
            break
        
    sent = ' '.join(chain)
    neg_list.append(sent)

output = np.asarray(neg_list)

total = 0

for j in range(len(output)):
    sentement = get_sent(output[j])
    total += sentement['neg']
    
average = total/len(output)
print("Average: ", average)

total = 0

for j in range(len(output)):
    sentement = get_sent(output[j])
    if(sentement['neg'] > 0.5):
        total += 1
        
error = 1 - (total/100)
print("Error: ", error)

np.savetxt('fake.out', output, fmt='%s')