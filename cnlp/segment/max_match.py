# -*- coding: utf-8 -*-
"""
File:    max_match
Author:  yulu04
Date:    2018/12/13
Desc:          
"""


# 最大正向匹配分词器
class MaxMatch(object):
    # 单词最大长度
    max_word_len = 5
    # 词典list
    word_dict = []

    # 初始化
    def __init__(self):
        self.max_word_len = 5

    # 设置词典
    def set_dict(self, word_dict):
        self.word_dict = word_dict

    # 分词
    def cut(self, text):
        words = []
        index = 0
        while index < len(text):
            for slice_len in range(self.max_word_len, 0, -1):
                slice_word = text[index: index + slice_len]
                if slice_word in self.word_dict or len(slice_word) <= 1:
                    words.append(slice_word)
                    index += len(slice_word)
                    break

        return words
