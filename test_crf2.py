# -*- coding: utf-8 -*-
"""
File:    test_crf
Author:  yulu04
Date:    2019/1/18
Desc:          
"""
from cnlp.model.crf import Crf
from cnlp.util.corpus import Corpus

template_file = './cnlp/conf/crf_templates2.txt'
file_path = './cnlp/data/trainCorpus.txt_utf8'

print('start read corpus')
corpus = Corpus.read_segment_corpus(file_path)
print('read corpus done')

crf = Crf()
print('start read feature template')
crf.read_feature_template(template_file)
print('read feature template done ')

crf.train(corpus, rate=0.01, iterations=50)

x = list('中华人民共和国成立了')
y = crf.predict(x)
print(y)
