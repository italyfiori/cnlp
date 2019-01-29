# -*- coding: utf-8 -*-
"""
File:    test_crf
Author:  yulu04
Date:    2019/1/18
Desc:          
"""

from cnlp.util.corpus import Corpus
from cnlp.model.crf import Crf

template_file = './cnlp/conf/crf_templates.txt'
train_file = 'cnlp/data/chunking_small/small_train.data'
test_file = 'cnlp/data/chunking_small/small_test.data'

train_data = Corpus.read_crf_corpus(train_file)

crf = Crf()
crf.read_feature_template(template_file)
crf.train(train_data, rate=0.01, iterations=50)

test_data = Corpus.read_crf_corpus(test_file)

total_count = 0
correct_count = 0
for X, Y in test_data:
    Yprime = crf.predict(X)
    for t in range(len(Y)):
        total_count += 1
        if Y[t] == Yprime[t]:
            correct_count += 1

print('Correct: %d' % correct_count)
print('Total: %d' % total_count)
print('Performance: %f' % (correct_count / total_count))
