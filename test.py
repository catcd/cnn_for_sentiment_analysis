if __name__ == '__main__':
    with open('/home/catcd/data/pretrained_w2v/fasttext/wiki.en.vocab', 'r', encoding='utf-8') as f:
        vocabf = f.read().split()

    vocabs = []
    for l in open('data/train_data/merge.txt'):
        vocabs.extend(l[2:].split())

    vocabs = list(set(vocabs))
    vocabs.sort()

    vocab = vocabf[:100000] + [i for i in vocabs if i in vocabf]

    vocab = list(set(vocab))

    with open('data/w2v/vocab.txt', 'w') as f:
        for w in vocab:
            f.write('{}\n'.format(w))

        f.write('$UNK$')
