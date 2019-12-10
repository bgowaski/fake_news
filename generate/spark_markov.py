from operator import add
import numpy as np
from pyspark import SparkContext

def get_parsed_args():
    import argparse
    parser = argparse.ArgumentParser(description='Script for generating fake '
                                     'news entry in Spark using Markov Chains.',
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input',
                        help='Input dataset.')
    parser.add_argument('maxwordlength',
			type=int,
                        help='Maximum word length of the fake news.')
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
def start_spark_context(nodes, name='Fake News Generator'):
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

class Frac(dict):
    def __getitem__(self, idx):
        if idx in self:
            return super().__getitem__(idx)
        else:
            return 0


def generate_fake_news(data, max_length):
    modd_data = data.map(lambda x: x + " SENTENCE_END")
    mrkv_pairs = modd_data.map(lambda x: x.split())\
                          .map(lambda x: [x[0].capitalize()] + x[1:])\
                          .flatMap(lambda x: [(x[i], x[i + 1]) for i in range(len(x) - 1)])\
                          .cache()
    first_element = mrkv_pairs.filter(lambda x: x[0] != "SENTENCE_END" and \
                                      x[1] != "SENTENCE_END" and x[0][0].isupper())\
                              .takeSample(False, 1)[0]
    fake_news = first_element[0]
    cur_wrd = first_element[1]
    for i in range(max_length):
        cur_pair = mrkv_pairs.filter(lambda x: x[0] == cur_wrd).takeSample(False, 1)[0]
        fake_news += (" " + cur_pair[0])
        cur_wrd = cur_pair[1]
        if (cur_wrd == "SENTENCE_END"):
            fake_news += "."
            break
    return fake_news

if __name__ == "__main__":
    from time import time
    args = get_parsed_args()
    sc = start_spark_context(args.sparknodes)
    start_time = time()
    data = read_input_data(sc, args.input, args.sparknodes).cache()
    fake_news = generate_fake_news(data, args.maxwordlength)
    print("Fake news: %s." % fake_news)
    end_time = time()
    print("Execution time from reading the input data to saving the computed "
          "data: %f", end_time - start_time)
