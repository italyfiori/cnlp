# -*- coding: utf-8 -*-
"""
File:    dict
Author:  yulu04
Date:    2018/12/13
Desc:          
"""


# 加载词典文件
def load(file_path):
    word_dict = set()
    with open(file_path, mode='r') as fo:
        for line in fo:
            word = line.strip().split(" ")[0]
            word_dict.add(word)

    return word_dict
