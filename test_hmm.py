# -*- coding: utf-8 -*-
"""
File:    test_hmm
Author:  yulu04
Date:    2019/1/29
Desc:          
"""

from cnlp.util.corpus import Corpus
import cnlp.model.hmm as hmm

file_path = './cnlp/data/trainCorpus.txt_utf8'
segment_corpus = Corpus.read_segment_corpus(file_path)
model = hmm.HMM()
model.train(['B', 'M', 'E', 'S'], segment_corpus)

text = '中华人民共和国成立了'
states = model.predict(list(text))
result = Corpus.states2segments(states, text)
print('/'.join(result))
