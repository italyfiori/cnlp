# -*- coding: utf-8 -*-
"""
File:    reverse_max_match
Author:  yulu04
Date:    2018/12/13
Desc:          
"""


# 最大逆向匹配分词器
class ReverseMaxMatch(object):
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

    def cut(self, text):
        words = []
        end_index = len(text)
        while end_index > 0:
            for slice_len in range(self.max_word_len, 0, -1):
                start_index = end_index - slice_len if end_index - slice_len >= 0 else 0
                slice_word = text[start_index: end_index]
                if slice_word in self.word_dict or len(slice_word) == 1:
                    words.append(slice_word)
                    end_index -= len(slice_word)
                    break

        words.reverse()
        return words
