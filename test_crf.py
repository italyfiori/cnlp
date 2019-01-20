# -*- coding: utf-8 -*-
"""
File:    test_crf
Author:  yulu04
Date:    2019/1/18
Desc:          
"""

from cnlp.util.corpus import Corpus
from cnlp.model.crf import Crf

train_file = 'cnlp/data/chunking_small/small_train.data'
test_file = 'cnlp/data/chunking_small/small_test.data'

train_data = Corpus.read_crf_corpus(train_file)
crf = Crf()
crf.train(train_data, rate=0.0005)

#
# def test(test_file):
#     test_data = Corpus.read_crf_corpus(test_file)
#
#     total_count = 0
#     correct_count = 0
#     for X, Y in test_data:
#         Yprime = crf.predict(X)
#         print(Yprime)
#         for t in range(len(Y)):
#             total_count += 1
#             if Y[t] == Yprime[t]:
#                 correct_count += 1
#
#     print('Correct: %d' % correct_count)
#     print('Total: %d' % total_count)
#     print('Performance: %f' % (correct_count / total_count))
#
#
# test(test_file)
