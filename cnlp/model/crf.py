# -*- coding: utf-8 -*-
"""
File:    my_crf
Author:  yulu04
Date:    2019/1/8
Desc:
"""
Y_START_LABEL = '*label_start*'
Y_NONE_LABEL = '*label_none*'

import numpy as np


class Crf(object):
    features_num = 0  # 用于生成feature_id
    features_dict = {}  # { x_feature : { y_feature : feature_id } }
    features_count = {}  # { feature_id : counts }
    labels_dict = {}  # 特征标签dict { y_feature: y_feature_id}

    weights = None
    features = None

    def __init__(self):
        pass

    # 训练模型
    def train(self, data):
        self.features_num = 0  # 用于生成feature_id
        self.features_dict = {}  # { x_feature : { y_feature : feature_id } }
        self.features_count = {}  # { feature_id : counts }
        self.labels_dict = {}  # 特征标签dict { y : y_id}
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
    def generate_X_trans_matrix(self, X):
        pass

    # 计算条件概率
    def generate_x_trans_matrix(self, weights, X, t):
        x_trans_matrix = np.zeros(len(self.labels_dict), len(self.labels_dict))
        x_feature_funcs = self.get_feature_funcs_from_dict(X, t)
        for (prev, y), feature_ids in x_feature_funcs.items():
            pass

    # 计算前向概率、后向概率和归一化因子
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

    def generate_feature_dict(self, data):
        """
        生成特征函数词典和标记的词典，用于辅助计算
        :param data:
        :return:
        """
        labels_num = 0
        for X, Y in data:
            for t in range(len(X)):
                x_features = self.get_x_features_from_template(X, t)
                y = Y[t]
                y_prev = Y[t - 1] if t > 0 else Y_START_LABEL
                y_features = [(y_prev, y), (Y_NONE_LABEL, y)]

                # 生成特征函数的词典, 形式为 { x_feature : { y_feature : feature_id } }
                self.fill_features_dict(x_features, y_features)

                # 生成状态词典, 形式为 { y : y_id }
                if y not in self.labels_dict:
                    self.labels_dict[y] = labels_num
                    labels_num += 1

    def data2features(self, data):
        """
        将观测序列数据转换成特征函数的表示形式, m个样本, 每个样本长度为T, 每个时刻有K个特征函数
        一组特征函数的形式为:  { (y_prev, y) : feature_ids }
        :param data:
        :return:
        """
        data_features = []
        for X, _ in data:
            data_features.append([self.get_feature_funcs_from_dict(X, t) for t in range(len(X))])

        return data_features

    def fill_features_dict(self, x_features, y_features):
        """
        生成保存特征函数的词典和特征函数的计数器
        词典形式为: { x_feature : { (y_prev, y) : feature_id } }
        计数器形式为: { feature_id : counts }
        :param x_features:
        :param y_features:
        :return:
        """
        for x_feature in x_features:
            if x_feature not in self.features_dict:
                self.features_dict[x_feature] = {}
            for y_feature in y_features:
                if y_feature not in self.features_dict[x_feature]:
                    # 首次出现的特征函数, 保存到词典, 计数为1
                    feature_id = self.features_num
                    self.features_dict[x_feature][y_feature] = feature_id
                    self.features_count[feature_id] = 1
                    self.features_num += 1
                else:
                    # 非首次出现的特征函数, 计数加1
                    feature_id = self.features_dict[x_feature][y_feature]
                    self.features_count[feature_id] += 1

    def get_feature_funcs_from_dict(self, X, t):
        """
        根据特征词典提取获观测序列X在t时刻对应的特征函数集合: { (y_prev, y) : feature_ids }
        :param X:
        :param t:
        :return:
        """
        feature_funcs = {}

        x_features = self.get_x_features_from_template(X, t)
        for x_feature in x_features:
            for (y_prev, y), feature_id in self.features_dict[x_feature].items():
                if (y_prev, y) not in feature_funcs:
                    feature_funcs[(y_prev, y)] = set()
                feature_funcs[(y_prev, y)].add(feature_id)
        return feature_funcs

    @staticmethod
    def get_x_features_from_template(X, t):
        """
        从特征函数模板中获取观测序列X在t时刻的x特征集合([x_feature1, x_feature2, ...])
        :param X:
        :param t:
        :return:
        """
        length = len(X)
        x_features = list()
        x_features.append('U[0]:%s' % X[t][0])
        x_features.append('POS_U[0]:%s' % X[t][1])
        if t < length - 1:
            x_features.append('U[+1]:%s' % (X[t + 1][0]))
            x_features.append('B[0]:%s %s' % (X[t][0], X[t + 1][0]))
            x_features.append('POS_U[1]:%s' % X[t + 1][1])
            x_features.append('POS_B[0]:%s %s' % (X[t][1], X[t + 1][1]))
            if t < length - 2:
                x_features.append('U[+2]:%s' % (X[t + 2][0]))
                x_features.append('POS_U[+2]:%s' % (X[t + 2][1]))
                x_features.append('POS_B[+1]:%s %s' % (X[t + 1][1], X[t + 2][1]))
                x_features.append('POS_T[0]:%s %s %s' % (X[t][1], X[t + 1][1], X[t + 2][1]))
        if t > 0:
            x_features.append('U[-1]:%s' % (X[t - 1][0]))
            x_features.append('B[-1]:%s %s' % (X[t - 1][0], X[t][0]))
            x_features.append('POS_U[-1]:%s' % (X[t - 1][1]))
            x_features.append('POS_B[-1]:%s %s' % (X[t - 1][1], X[t][1]))
            if t < length - 1:
                x_features.append('POS_T[-1]:%s %s %s' % (X[t - 1][1], X[t][1], X[t + 1][1]))
            if t > 1:
                x_features.append('U[-2]:%s' % (X[t - 2][0]))
                x_features.append('POS_U[-2]:%s' % (X[t - 2][1]))
                x_features.append('POS_B[-2]:%s %s' % (X[t - 2][1], X[t - 1][1]))
                x_features.append('POS_T[-2]:%s %s %s' % (X[t - 2][1], X[t - 1][1], X[t][1]))

        return x_features
