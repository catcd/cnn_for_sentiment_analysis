import numpy as np
from sklearn.utils import shuffle
from collections import Counter

np.random.seed(13)


def process_data(data_name, vocab_words):
    data_words = []
    labels = []

    with open('{}.txt'.format(data_name), 'r') as f:
        for line in f:
            l, ws = line.strip().split(maxsplit=1)
            data_words.append(ws.split())
            labels.append(int(l))

    words = []
    for i in data_words:
        ws = [vocab_words[w] if w in vocab_words else vocab_words['$UNK$'] for w in i]
        words.append(ws)

    return Dataset(words, labels)


class Dataset:
    def __init__(self, words, labels=None):
        self.words = words
        self.labels = labels

    def shuffle_data(self):
        if self.labels is not None:
            (
                self.labels,
                self.words,
            ) = shuffle(
                self.labels,
                self.words,
            )
        else:
            self.words = shuffle(self.words)

    def one_vs_nine(self):
        c = Counter(self.labels)
        print('shape of data: {}'.format({k: c[k] for k in c}))
        num_of_example = len(self.labels)
        indicates = np.random.choice(num_of_example, num_of_example // 10, replace=False)

        one_labels = [v for i, v in enumerate(self.labels) if i in indicates]
        one_words = [v for i, v in enumerate(self.words) if i in indicates]
        c = Counter(one_labels)
        print('shape of 10% data: {}'.format({k: c[k] for k in c}))

        nine_labels = [v for i, v in enumerate(self.labels) if i not in indicates]
        nine_words = [v for i, v in enumerate(self.words) if i not in indicates]
        c = Counter(nine_labels)
        print('shape of 90% data: {}'.format({k: c[k] for k in c}))

        return Dataset(one_words, one_labels), Dataset(nine_words, nine_labels)
