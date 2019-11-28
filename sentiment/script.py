"""
" Authors: Benjamin Gowaski, Matin Raayai Ardakani
" This script reads in a news dataset to an RDD in Spark, does sentiment analysis
" it and outputs the results in a pre-defined format.
"""
from pyspark import SparkContext
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
sc = SparkContext('local[2]')
data = sc.textFile('abcnews-date-text.csv')
def get_sent(sentence):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(sentence)
out = data.map(get_sent) 
