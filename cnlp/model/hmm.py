# -*- coding: utf-8 -*-

from functools import reduce
import pickle


class HMM(object):
    states = []
    init_probs = {}
    trans_probs = {}
    emit_probs = {}

    def __init__(self):
        pass

    def train(self, states, train_seq):
        """
        训练HMM模型
        :param states: 模型状态集合
        :param train_seq: 训练集, 每一条数据包含状态序列和观测序列
        :return:
        """
        self.init_params(states)
        for observe_seq, state_seq in train_seq:
            assert len(state_seq) == len(observe_seq)
            for i in range(len(state_seq)):
                cur_state = state_seq[i]
                cur_observe = observe_seq[i]
                assert cur_state in states
                if i == 0:
                    # 状态初始概率计数
                    self.init_probs[cur_state] = self.init_probs.get(cur_state, 0) + 1
                else:
                    # 状态转移概率计数
                    prev_state = state_seq[i - 1]
                    self.trans_probs[prev_state][cur_state] = self.trans_probs[prev_state].get(cur_state, 0) + 1
                # 发射概率计数
                self.emit_probs[cur_state][cur_observe] = self.emit_probs[cur_state].get(cur_observe, 0) + 1

        # 状态初始概率
        init_states_count = sum(self.init_probs.values())
        self.init_probs = {state: float(count) / init_states_count for state, count in self.init_probs.items()}

        # 状态转移概率
        for state_i in self.trans_probs:
            state_i_trans_probs = self.trans_probs[state_i]
            state_i_trans_count = sum(state_i_trans_probs.values())

            self.trans_probs[state_i] = {
                state_j: 0 if state_i_trans_count == 0 else float(state_j_count) / state_i_trans_count
                for state_j, state_j_count in state_i_trans_probs.items()}

        # 发射概率
        for state_i in self.emit_probs:
            state_i_emit_probs = self.emit_probs[state_i]
            state_i_emit_count = sum(state_i_emit_probs.values())
            self.emit_probs[state_i] = {observe: float(observe_count) / state_i_emit_count
                                        for observe, observe_count in state_i_emit_probs.items()}

    def init_params(self, states):
        self.states = states
        self.init_probs = {state: 0 for state in states}
        self.trans_probs = {state_i: {state_j: 0 for state_j in states} for state_i in states}
        self.emit_probs = {state: {} for state in states}

    def viterbi(self, observe_seq, default_state=None, default_end_states=None):
        assert len(observe_seq) >= 1
        assert default_state is None or default_state in self.states
        assert default_end_states is None or len(set(default_end_states).difference(set(self.states))) == 0

        # 保存状态路径的最大值概率值
        viterbi_matrix = [{}]
        # 保存最优状态路径
        viterbi_path = {}

        # 初始状态
        observe_value = observe_seq[0]
        for state in self.states:
            viterbi_matrix[0][state] = self.init_probs[state] * self.emit_probs[state].get(observe_value, 0)
            viterbi_path[state] = [state]

        for i in range(1, len(observe_seq)):
            # 防止路径联合概率过小
            if sum(viterbi_matrix[-1].values()) < 1e-100:
                viterbi_matrix[-1] = {state: prob * 1e100 for state, prob in viterbi_matrix[-1].items()}

            viterbi_matrix.append({})
            observe_value = observe_seq[i]

            # 所有状态都没有观测到observe_value
            not_observe = all([observe_value not in self.emit_probs[state] for state in self.states])

            viterbi_path_tmp = {}
            for cur_state in self.states:
                if not_observe and default_state is not None:
                    # 1: 没有观测到observe_value, 且有默认状态时, 使用默认状态
                    emit_prob = 1 if cur_state == default_state else 0
                else:
                    # 2: 没有观测到observe_value, 且没有默认状态时, 假设所有状态的发射概率都相同且为1
                    # 3: 观测到observe_value时, 使用发射概率
                    emit_prob = 1 if not_observe else self.emit_probs[cur_state].get(observe_value, 0)

                cur_prob, prev_state = max(
                    [(viterbi_matrix[i - 1][prev_state] * self.trans_probs[prev_state].get(cur_state, 0) * emit_prob,
                      prev_state) for prev_state in self.states])

                viterbi_matrix[i][cur_state] = cur_prob
                viterbi_path_tmp[cur_state] = viterbi_path[prev_state] + [cur_state]
            viterbi_path = viterbi_path_tmp

        last_prob, last_state = max([(viterbi_matrix[-1][state], state) for state in self.states])
        return viterbi_path[last_state]

    def load_mode(self, model_path):
        model_params = pickle.load(open(model_path, 'rb'))
        self.states = model_params.states
        self.init_probs = model_params.init_probs
        self.trans_probs = model_params.trans_probs
        self.emit_probs = model_params.emit_probs

    def save_mode(self, model_path):
        model_params = {
            'states': self.states,
            'init_probs': self.init_probs,
            'trans_probs': self.trans_probs,
            'emit_probs': self.emit_probs,
        }
        pickle.dump(model_params, open(model_path, 'wb'))
