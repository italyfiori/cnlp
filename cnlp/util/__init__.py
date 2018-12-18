# -*- coding: utf-8 -*-
"""
File:    __init__
Author:  yulu04
Date:    2018/12/13
Desc:          
"""

import time


def micro_time():
    return int(round(time.time() * 1000))
