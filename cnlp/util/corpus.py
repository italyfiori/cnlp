# -*- coding: utf-8 -*-

from functools import reduce


class Corpus(object):
    @staticmethod
    def word2states(word):
        if len(word) == 1:
            return ['S']
        else:
            return ['B'] + ['M'] * (len(word) - 2) + ['E']

    @staticmethod
    def states2segments(states, text):
        assert len(set(states).intersection(set(['B', 'E', 'M', 'S']))) == len(set(states))

        text = list(text)
        segments = []

        start_pos = 0
        for i in range(len(states)):
            state = states[i]
            if state in ['E', 'S']:
                word = text[start_pos: i + 1]
                segments.append(''.join(word))
                start_pos = i + 1

        if start_pos < len(states):
            segments.append(''.join(text[start_pos:]))
        return segments

    @staticmethod
    def read_segment_corpus(file_path):
        print(file_path)
        with open(file_path) as fo:
            corpus = []
            for line in fo:
                # line = line.decode('utf8')
                words = [word.strip('') for word in line.strip().split(' ')]
                labels = [Corpus.word2states(word) for word in words]

                labels = reduce(lambda a, b: a + b, labels)
                words = [list(word) for word in words]
                words = reduce(lambda a, b: a + b, words)
                corpus.append((words, labels))

            return corpus

    @staticmethod
    def read_crf_corpus(filename):
        """
        Read a corpus file with a format used in CoNLL.
        """
        data = list()
        data_string_list = list(open(filename))

        cols = 0
        X = list()
        Y = list()
        for data_string in data_string_list:
            words = data_string.strip().split()

            if len(words) is not 0:
                # 保证序列的每一个元素的列数相同
                if cols is 0:
                    cols = len(words)
                assert len(words) == cols

                # 非空行时取数据
                X.append(words[:-1])
                Y.append(words[-1])
            else:
                # 遇到空行时将前一个序列插入集合
                if len(X) != 0:
                    data.append((X, Y))

                # 新序列置空
                X = list()
                Y = list()

        # 最后一个序列插入集合
        if len(X) > 0:
            data.append((X, Y))

        # data.shape: m * 2 * Tm
        return data
