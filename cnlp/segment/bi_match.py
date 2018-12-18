# -*- coding: utf-8 -*-
"""
File:    bi_match
Author:  yulu04
Date:    2018/12/13
Desc:          
"""

from . import max_match as mat_match
from . import reverse_max_match as reverse_max_match


# 双向分词器
class BiMatch(object):
    def __init__(self):
        # 加载正向和逆向分词器
        self.mm = mat_match.MaxMatch()
        self.rm = reverse_max_match.ReverseMaxMatch()

    # 设置分词器词典
    def set_dict(self, word_dict):
        self.mm.set_dict(word_dict)
        self.rm.set_dict(word_dict)

    # 分词
    def cut(self, text):
        mm_words = self.mm.cut(text)
        rm_words = self.rm.cut(text)
        # 正反切分结果相同
        if mm_words == rm_words:
            return mm_words

        # 正反切分单词数不同时，返回切分单词数少的结果
        if len(mm_words) != len(rm_words):
            return mm_words if len(mm_words) < len(rm_words) else rm_words

        mm_single_word_num = len(list(filter(lambda word: len(word) == 1, mm_words)))
        rm_single_word_num = len(list(filter(lambda word: len(word) == 1, rm_words)))

        # 正反切分单词数相同时，返回单字单词少的切分结果
        if mm_single_word_num != rm_single_word_num:
            return mm_words if mm_single_word_num < rm_single_word_num else rm_words
        # 正反切分单词数相同且单字单词数也相同时，返回最大逆向切分
        else:
            return rm_single_word_num
