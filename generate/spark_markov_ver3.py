def generate_fake_news(data,
                       min_num_entries,
                       min_length,
                       max_length,
                       num_parts,
                       count_timeout=1,
                       starting_comp_entries=1000):
    from numpy.random import choice
    from operator import add
    from time import time
    # Will incrementally save output as we pass min_length.
    print(time())
    start_time = time()
    output = []
    num_entries_remaining = min_num_entries
    modd_data = data.map(lambda x: x + " SENTENCE_END")
    mrkv_pairs = modd_data.map(lambda x: x.split())\
                          .map(lambda x: [x[0].capitalize()] + x[1:])\
                          .flatMap(lambda x: [(x[i], x[i + 1]) for i in range(len(x) - 1)])\
                          .partitionBy(num_parts)\
                          .cache()
    revrsed_mrkv_pairs = mrkv_pairs.map(lambda x: (x[1], x[0]))\
                                   .filter(lambda x: x[1][0].isupper() and \
                                           x[1] != "SENTENCE_END")\
                                   .cache()

    cur_approx_count = revrsed_mrkv_pairs.countApprox(count_timeout)
    cur_entries = revrsed_mrkv_pairs.sample(False,
                                            float(starting_comp_entries) / cur_approx_count)\
                                    .map(lambda x: (x[0], (x[1], x[1] + " " + x[0])))\
                                    .partitionBy(num_parts)\
                                    .cache()
    """
    Transforms each entry of the matrix to a representation such that each
    path can be sampled independently without much computational overhead.
    """
    def hash_transform(x):
        sent = x[0][1][0][1] + " " + x[0][1][1]
        return ((hash(x[0][1][0][1]), [x[1]]), (x[0][1][1], (x[0][0], sent)))

    # For loop starts here:----------------------------------------------------#
    for i in range(max_length):
        cur_entries = cur_entries.join(mrkv_pairs)\
                                 .zipWithUniqueId()\
                                 .map(hash_transform)
        smpl_rdd = cur_entries.keys()\
                              .reduceByKey(add)\
                              .mapValues(choice)\
                              .values()\
                              .map(lambda x: (x, True))\
                              .partitionBy(num_parts, lambda x: x[0])
        cur_entries = cur_entries.map(lambda x: (x[0][1][0], x[1]))\
                                 .partitionBy(num_parts, lambda x: x[0])\
                                 .join(smpl_rdd)\
                                 .map(lambda x: x[1][0]).cache()
        # Approximate the count. We don't want to waste too much computational
        # power here.
        cur_total_entries = cur_entries.countApprox(1)

        # Clip the number of entries to the starting computational entries.
        if cur_total_entries > starting_comp_entries:
            cur_entries = cur_entries.sample(False,
                                             float(starting_comp_entries) / cur_total_entries)\
                                     .cache()
        # We can extract news entries that have reached the end of the sentence
        # if we have met the minimum required word length.
        if i >= min_length:
            extcted_out = cur_entries.filter(lambda x: x[1][1].split()[-1] == "SENTENCE_END")\
                                     .map(lambda x: x[1][1])\
                                     .collect()
            num_entries_remaining -= len(extcted_out)
            output.extend(extcted_out)
        # If the number of entries have reached bellow the specified amount,
        # break out of the loop.
        if cur_total_entries < num_entries_remaining:
            break
        # Repartition for the next loop.
        cur_entries = cur_entries.partitionBy(num_parts)
    output.extend(cur_entries.map(lambda x: x[1][1]).collect())
    end_time = time()
    print(time())
    print("Total amount of time: %f" % (end_time - start_time))
    return output
