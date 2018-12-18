# -*- coding: utf-8 -*-
"""
File:    __init__.py
Author:  yulu04
Date:    2018/12/13
Desc:          
"""

import os.path as path

# 基础目录
cnlp_dir = path.dirname(path.dirname(__file__))
data_dir = path.join(cnlp_dir, 'data')
conf_dir = path.join(cnlp_dir, 'conf')
util_dir = path.join(cnlp_dir, 'util')

# 词典文件
dict_file = path.join(data_dir, 'dict', 'dict.txt')
