# -*- coding: utf-8 -*-
"""
File:    crf_feature
Author:  yulu04
Date:    2019/1/8
Desc:
"""

Y_START_LABEL = 'start'
Y_EMPTY_LABEL = 'empty'


class Feature(object):

    def __init__(self):
        self.features_num = 0  # 用于生成feature_id
        self.features_dict = {}  # { x_feature : { y_feature : feature_id } }
        self.features_count = {}  # { feature_id : counts }
        self.labels_dict = {}  # 特征标签dict

    # 序列数据转化成特征函数形式，并生成保存特征函数的词典和特征函数的计数器
    def train_data2features(self, data):
        labels_num = 0
        for X, Y in data:
            for t in range(len(X)):
                x_features = self.get_x_features_from_template(X, t)
                y = Y[t]
                y_prev = Y[t - 1] if t > 0 else Y_START_LABEL
                y_features = [(Y_EMPTY_LABEL, y), (y_prev, y)]
                self.generate_feature_dict(x_features, y_features)

                if y not in self.labels_dict:
                    self.labels_dict[y] = labels_num
                    labels_num += 1

    def generate_feature_dict(self, x_features, y_features):
        """
        生成保存特征函数的词典和特征函数的计数器
        词典形式为: { x_feature : { y_feature : feature_id } }
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

    # 从特征词典中获取xt的特征函数集合 ([((y_prev, y), feature_id), ... ])
    def get_x_feature_funcs_from_dict(self, X, t):
        x_features = self.get_x_features_from_template(X, t)
        return x_features

    # 从特征函数模板中获取xt的x特征集合([x_feature1, x_feature2, ...])
    def get_x_features_from_template(self, X, t):
        return []
