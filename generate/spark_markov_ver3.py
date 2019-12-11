def generate_fake_news(data,
                       num_entries,
                       max_length,
                       num_parts,
                       count_timeout=1,
                       max_comp_entries=100):
     from numpy.random import choice
     from operator import add
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
     aprx_count = revrsed_mrkv_pairs.countApprox(count_timeout)
     cur_matrix = revrsed_mrkv_pairs.sample(False, float(max_comp_entries) / aprx_count)\
                                    .map(lambda x: (x[0], (x[1], x[1] + " " + x[0])))\
                                    .partitionBy(num_parts)\
                                    .cache()
     def hash_transform(x):
         sent = x[0][1][0][1] + " " + x[0][1][1]
         return ((hash(x[0][1][0][1]), [x[1]]), (x[0][1][1], (x[0][0], sent)))
     for i in range(max_length):
         cur_matrix = cur_matrix.join(mrkv_pairs)\
                                .zipWithUniqueId()\
                                .map(hash_transform)
         smpl_rdd = cur_matrix.keys()\
                              .reduceByKey(add)\
                              .mapValues(choice)\
                              .values()\
                              .map(lambda x: (x, True))\
                              .partitionBy(num_parts)
         cur_matrix = cur_matrix.map(lambda x: (x[0][1][0], x[1]))\
                                .partitionBy(num_parts)\
                                .join(smpl_rdd)\
                                .map(lambda x: x[1][0]).cache()
         total_entries = cur_matrix.countApprox(count_timeout)
         if total_entries > max_comp_entries:
             cur_matrix = cur_matrix.sample(False, float(max_comp_entries) / total_entries)
         cur_matrix = cur_matrix.partitionBy(num_parts)\
                                .cache()
         print(cur_matrix.count())
         print("\n\n")
     return cur_matrix.collect()
 

