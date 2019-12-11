def generate_fake_news(data,
                       num_entries,
                       max_length,
                       num_parts,
                       count_timeout=1,
                       max_comp_entries=100):
    modd_data = data.map(lambda x: x + " SENTENCE_END")
    mrkv_pairs = modd_data.map(lambda x: x.split())\
                          .map(lambda x: [x[0].capitalize()] + x[1:])\
                          .flatMap(lambda x: [(x[i], x[i + 1]) for i in range(len(x) - 1)])\
                          .partitionBy(num_parts)\
                          .cache()
    revrsed_mrkv_pairs = mrkv_pairs.map(lambda x: (x[1], x[0]))\
                                   .cache()
    aprx_count = revrsed_mrkv_pairs.countApprox(count_timeout)
    cur_matrix = revrsed_mrkv_pairs.sample(False, float(max_comp_entries) / aprx_count)\
                                   .map(lambda x: (x[0], (x[1], [x[1], x[0]])))\
                                   .partitionBy(num_parts)\
                                   .cache()
    for i in range(max_length):
        cur_matrix = cur_matrix.join(mrkv_pairs)\
                               .map(lambda x: (x[1][1], (x[0], x[1][0][1] + [x[1][1]])))\
                               .cache()
        cur_aprx_count = cur_matrix.countApprox(count_timeout)
        cur_matrix = cur_matrix.sample(False, float(max_comp_entries) / cur_aprx_count)\
                               .partitionBy(num_parts)\
                               .cache()
    return cur_matrix.map(lambda x: ' '.join(x[1][1])).takeSample(False, num_entries)
