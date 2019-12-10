"""
Authors: Ben Gowaski, Siddith Gagger, Matin Raayai Ardakani
Script for calculating the sentiment of each news entry of a dataset using
Apache Spark and various text analytics Python libraries and filtering out the
ones with the desired sentiment polarity score.
"""

"""
Parses the command line arguments and returns them.
:return parsed commandline args.
"""
def get_parsed_args():
    import argparse
    parser = argparse.ArgumentParser(description='Script for calculating the sentiment of '
                                     'each news entry of a dataset using Apache'
                                     'Spark and various text analytics Python'
                                     'libraries.',
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input',
                        help='Input dataset.')
    parser.add_argument('output',
                        help='Path for the output dataset.')
    parser.add_argument('sentimentlibrary',
                        choices=['vader', 'textblob'],
                        help='The text analyzing library used for sentiment '
                        'calculations,')
    parser.add_argument('lowerbound',
                        type=float,
                        default=-1.0,
                        help='The lower bound for the sentiment scores of the '
                             'news extracted.')
    parser.add_argument('upperbound',
                        type=float,
                        default=1.0,
                        help='The upper bound for the sentiment scores of the '
                             'news extracted.')
    parser.add_argument('--sparknodes',
                        type=int,
                        default=20,
                        help="Number of Spark CPU workers.")
    return parser.parse_args()


"""
Starts a SparkContext locally with the given number of nodes.
:param nodes Number of nodes (CPUs) used in the context.
:param name Name of the SparkContext.
:return Initializd SparkContext object.
"""
def start_spark_context(nodes, name='Sentiment Analysis'):
    from pyspark import SparkContext
    print("Starting Spark Context with %d nodes." % nodes)
    return SparkContext('local[%d]' % nodes, name)

"""
Reads in a news dataset from the given path and creates a new rdd in the given
SparkContext with the specified number of partitions containing the dataset.
:param spark_context the SparkContext the dataset will reside in.
:param path where the dataset resides on disk.
:param num_partitions Number of partitions for the data RDD.
:param an RDD in the given SparkContext with the specified number of partitions
with each element being a single entry of the news dataset.
"""
def read_input_data(spark_context, path, num_partitions):
    print("Reading data from %s." % path)
    in_data = spark_context.textFile(path).repartition(num_partitions)
    print("Number of data partitions: %d" % in_data.getNumPartitions())
    return in_data

"""
Calculates the sentiment polarity score of a single sentence using the VADER
text analysis library.
:param sentence a string containing the sentence.
:return the sentiment polarity score of the sentence as a float.
"""
def calculate_sentence_sentiment_vader(sentence):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer().polarity_scores(sentence)['compound']

"""
Calculates the sentiment polarity score of a single sentence using the textblob
text analysis library.
:param sentence a string containing the sentence.
:return the sentiment polarity score of the sentence as a float.
"""
def calculate_sentence_sentiment_textblob(sentence):
    from textblob import TextBlob
    return TextBlob(sentence).sentiment.polarity


"""
Calculates the polarity score of each news entry in an RDD using the specified 
text analysis library and returns the result in a new RDD.
:param data the rdd with each entry being a single news entry.
:param sentiment_library the text analysis library to be used to calculate the 
sentiment of each news entry.
:return a new key-value pair RDD with the format of (news, polarity_score).
"""
def calculate_rdd_sentiment(data, sentiment_library):
    if sentiment_library == 'vader':
        print("Using VADER for sentiment calculations.")
        analyzer_func = calculate_sentence_sentiment_vader
    elif sentiment_library == 'textblob;':
        print("Using TextBlob for sentiment calculations.")
        analyzer_func = calculate_sentence_sentiment_textblob
    else:
        raise NotImplementedError("The given sentiment analyzing library is not"
                                  "implemented")
    return data.map(lambda sentence: (sentence, analyzer_func(sentence)))

"""
Extracts an interval of sentiment from a key-value pair RDD containing news 
entries and their sentiment scores.
:param sntmnt_rdd a key-value pair RDD with the format of (news, polarity_score)
:param lowerbound the lower bound of the score interval.
:param upperbound the upper bound of the score interval.
:return a new key-value pair RDD containing news entries and their sentiment scores,
where lowerbound <= sentiment_score <= upperbound for all the news entries.
"""
def fltr_intvl_from_sntmnt_rdd(sntmnt_rdd, lowerbound, upperbound):
    if lowerbound > upperbound:
        raise ValueError("Lower bound is greater than the upper bound.")
    print("Extracting news entries with %f <= sentiment_score <= %f" % (lowerbound, upperbound))
    return sntmnt_rdd.filter(lambda x: lowerbound <= x[1] <= upperbound)



def save_results(calculated_stmnt_rdd, path):
    print("Saving results to %s." % path)
    calculated_stmnt_rdd.map(lambda x: x[0]).saveAsTextFile(path)

if __name__ == "__main__":
    from time import time
    args = get_parsed_args()
    sc = start_spark_context(args.sparknodes)
    start_time = time()
    data = read_input_data(sc, args.input, args.sparknodes)
    sntmnt_rdd = calculate_rdd_sentiment(data, args.sentimentlibrary)
    sntmnt_rdd = fltr_intvl_from_sntmnt_rdd(sntmnt_rdd, args.lowerbound, args.upperbound)
    save_results(sntmnt_rdd, args.output)
    end_time = time()
    print("Execution time from reading the input data to saving the computed "
          "data: %f" % (end_time - start_time))
