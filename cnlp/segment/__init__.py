# -*- coding: utf-8 -*-
"""
File:    __init__
Author:  yulu04
Date:    2018/12/13
Desc:
"""

from . import bi_match
import cnlp.conf as conf
import cnlp.util.dict_processor as dict_processor

# 双向分词class
bi_match_seg = bi_match.BiMatch()


# 初始化, 加载词典
def init():
    word_dict = dict_processor.load(conf.dict_file)
    bi_match_seg.set_dict(word_dict)


# 分词
def cut(text):
    if type(text) == str:
        # text = text.decode('utf8')
        pass
    # assert type(text) == unicode

    words = bi_match_seg.cut(text)
    return [word for word in words]
