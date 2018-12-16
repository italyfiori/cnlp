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
