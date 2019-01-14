# -*- coding: utf-8 -*-
"""
File:    my_crf
Author:  yulu04
Date:    2019/1/8
Desc:
"""

from feature import Feature


class Crf(object):
    def __init__(self):
        self.feature = Feature()
        self.weights = None
        self.features = None

    # 训练模型
    def train(self, data):
        pass

    # 预测标记
    def predict(self, X):
        pass

    # 计算似然函数
    def calc_likelihood(self, weights, data):
        pass

    # 计算梯度
    def calc_gradient(self, weights, data):
        pass

    # 计算转移概率矩阵
    def generate_trans_matrix(self, X):
        pass

    # 计算转移概率
    def generate_trans_prob(self, X, t):
        pass

    # 计算前向概率、后向概率和归一化银子
    def forward_backward(self, X):
        pass

    # 生成特征函数
    def generate_features(self, data):
        pass

    # 保存模型
    def save_model(self):
        pass

    # 计算模型
    def load_model(self):
        pass
