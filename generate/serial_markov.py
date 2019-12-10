def get_parsed_args():
    import argparse
    parser = argparse.ArgumentParser(description='Script for generating fake '
                                     'news entry serially using Markov Chains.',
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input',
                        help='Input dataset.')
    parser.add_argument('maxwordlength',
                        type=int,
                        help='Maximum word length of the fake news.')
    parser.add_argument('numentries',
                        type=int,
                        help='Number of fake news entries')
    return parser.parse_args()

def read_words_from_input_data(path):
    input_file = open(path, 'r')
    all_news = input_file.readlines()
    data = [wrd for news in all_news for wrd in news]
    input_file.close()
    return data

def make_pairs(corpus):
        for i in range(len(corpus) - 1):
            yield (corpus[i], corpus[i + 1])
    

def generate_fake_news(data, length, num):
    pairs = make_pairs(data)
    word_dict = {}
    for word_1, word_2 in pairs:
        if word_1 in word_dict.keys():
            word_dict[word_1].append(word_2)
        else:
            word_dict[word_1] = [word_2]
    fk_news_list = []
    for k in range(num):
        first_word = choice(data)
        chain = [first_word]

        for i in range(length):
            try:
                chain.append(choice(word_dict[chain[-1]]))
            except KeyError:
                break
        fk_news_list.append(' '.join(chain))
    return fk_news_list




if __name__ == "__main__":
    from time import time
    from numpy.random import choice
    args = get_parsed_args()
    start_time = time()
    data = read_words_from_input_data(args.input)
    fake_news_list = generate_fake_news(data, args.maxwordlength, args.numentries)
    print("Fake news: %s." % fake_news_list)
    end_time = time()
    print("Execution time from reading the input data to saving the computed "
          "data: %f", end_time - start_time)

    
