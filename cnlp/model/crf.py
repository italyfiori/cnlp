# -*- coding: utf-8 -*-
"""
File:    my_crf
Author:  yulu04
Date:    2019/1/8
Desc:
"""
import numpy as np
from math import *

Y_START_LABEL = '|-'
Y_START_LABEL_INDEX = 0
Y_NONE_LABEL_INDEX = -1


class Crf(object):
    features_dict = None  # { x_feature : { y_feature : feature_id } }
    features_counts = None  # np array { feature_id : counts }
    labels_dict = None  # 特征标签dict { y_feature: y_feature_id}
    weights = None
    features = None

    def __init__(self):
        pass

    # 训练模型
    def train(self, data):
        self.features_dict = {}  # { x_feature : { y_feature : feature_id } }
        self.features_counts = None  # { feature_id : counts }
        self.labels_dict = {}  # 特征标签dict { y : y_id}

        self.generate_feature_dict(data)

        self.weights = np.zeros((len(self.features_counts), 1))

    # 预测标记
    def predict(self, X):
        pass

    # 计算似然函数
    def calc_likelihood_and_gradient(self, weights, data, squared_sigma):
        feature_expects = np.zeros((len(self.features_counts)))

        total_Z = 0

        for X, _ in data:
            # 计算转移概率矩阵
            trans_matrix_list = self.generate_seq_trans_matrix(weights, X)
            # 计算前向后向概率
            alpha_matrix, beta_matrix, Z = self.forward_backward(X, trans_matrix_list)
            # 归一化因子加和
            total_Z += log(Z)

            for t in range(len(X)):

                feature_funcs = self.get_feature_funcs_from_dict(X, t)
                for (y_prev, y), feature_ids in feature_funcs.items():
                    # 计算概率 Pi(y_prev, y|X)
                    feature_prob = alpha_matrix[t, y_prev] * trans_matrix_list[t][y_prev, y] * \
                                   beta_matrix[t + 1, y] / Z
                    for feature_id in feature_ids:
                        feature_expects[feature_id] += feature_prob

        # 似然函数
        likelihood = np.dot(self.features_counts, weights) - total_Z + \
                     np.sum(np.dot(weights, weights)) / (squared_sigma * 2)

        # 计算梯度
        gradient = self.features_counts - feature_expects - weights / squared_sigma

        return likelihood, gradient

    def generate_seq_trans_matrix(self, weights, X):
        """
        计算X在所有时刻的转移概率矩阵 M1(y1_prev, y1|X), M2(y2_prev, y2|X), ......
        :param weights:
        :param X:
        :return: T * labels_num * labels_num
        """
        trans_matrix_list = []

        for t in range(len(X)):
            trans_matrix = self.generate_trans_matrix(weights, X, t)
            trans_matrix_list.append(trans_matrix)

        return trans_matrix_list

    def generate_trans_matrix(self, weights, X, t):
        """
        计算转移概率矩阵: Mi(yi_prev, yi|X)
        :param weights:
        :param X:
        :param t:
        :return: labels_num * labels_num
        """
        labels_num = len(self.labels_dict)
        trans_matrix = np.zeros(labels_num, labels_num)

        feature_funcs = self.get_feature_funcs_from_dict(X, t)
        for (y_prev, y), feature_ids in feature_funcs.items():
            # 特征函数与权重的内积
            weights_sum = [weights[feature_id] for feature_id in feature_ids]
            if y_prev != Y_NONE_LABEL_INDEX:
                trans_matrix[y_prev, y] = weights_sum

        if t == 0:
            # 起始时刻， y_prev都为START状态， 其他状态的转移概率为0
            trans_matrix[Y_START_LABEL_INDEX + 1:] = 0
        else:
            # 非起始时刻, y_prev和y都不能为START状态，相应的状态转移概率为0
            trans_matrix[:, Y_START_LABEL_INDEX] = 0
            trans_matrix[Y_START_LABEL_INDEX, :] = 0

        return np.exp(trans_matrix)

    # 计算前向概率、后向概率和归一化因子
    def forward_backward(self, X, trans_matrix_list):
        matrix_len = len(X) + 1
        alpha_matrix = np.zeros((matrix_len, len(self.labels_dict)))  # T + 1 * labels_num
        beta_matrix = np.zeros((matrix_len, len(self.labels_dict)))  # T + 1 * labels_num

        alpha_matrix[0][Y_START_LABEL_INDEX] = 1.0
        for t in range(1, matrix_len):
            # alpha_matrix[t].shape: labels_num * 1,
            # trans_matrix_list.shape: labels_num * labels *num
            alpha_matrix[t] = np.dot(alpha_matrix[t - 1].T, trans_matrix_list[t - 1])

        beta_matrix[-1] = 1.0
        for t in range(matrix_len - 2, -1, -1):
            beta_matrix[t] = np.dot(trans_matrix_list[t], beta_matrix[t + 1])

        Z = sum(alpha_matrix[-1])
        return alpha_matrix, beta_matrix, Z

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
        self.labels_dict[Y_START_LABEL] = Y_START_LABEL_INDEX

        for X, Y in data:
            for t in range(len(X)):
                x_features = self.get_x_features_from_template(X, t)
                y = Y[t]
                y_prev = Y[t - 1] if t > 0 else Y_START_LABEL

                # 生成状态词典, 形式为 { y : y_id }
                if y not in self.labels_dict:
                    self.labels_dict[y] = len(self.labels_dict)

                y_idx = self.labels_dict[y]
                y_prev_idx = self.labels_dict[y_prev]
                y_features = [(y_prev_idx, y_idx), (Y_NONE_LABEL_INDEX, y_idx)]

                # 生成特征函数的词典, 形式为 { x_feature : { y_feature : feature_id } }
                self.fill_features_dict(x_features, y_features)

    def data2features(self, data):
        """
        将观测序列数据转换成特征函数的表示形式。 m个样本, 每个样本长度为T, 每个时刻有K个特征函数
        特征函数的形式为:  { (y_prev, y) : feature_ids }
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
                    feature_id = len(self.features_counts)
                    self.features_dict[x_feature][y_feature] = feature_id
                    self.features_counts[feature_id] = 1
                else:
                    # 非首次出现的特征函数, 计数加1
                    feature_id = self.features_dict[x_feature][y_feature]
                    self.features_counts[feature_id] += 1

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
