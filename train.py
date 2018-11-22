import argparse
from sklearn.metrics import precision_recall_fscore_support

from utils import load_vocab, get_trimmed_w2v_vectors, Timer
from dataset import process_data
from model import CnnModel
import config

parser = argparse.ArgumentParser(description='Train Multi-region-size CNN for Sentiment Analysis')

parser.add_argument('-train', help='Train data', type=str, required=True)
parser.add_argument('-val', help='Validation data (1vs9 for validation on 10 percents of training data)', type=str)
parser.add_argument('-test', help='Test data', type=str)

parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
parser.add_argument('-p', help='Patience of early stop (0 for ignore early stop)', type=int, default=10)
parser.add_argument('-b', help='Batch size', type=int, default=128)

parser.add_argument('-pre', help='Pre-trained weight', type=str)
parser.add_argument('-name', help='Saved model name', type=str, required=True)

opt = parser.parse_args()
print('Running opt: {}'.format(opt))


def main():
    timer = Timer()
    timer.start('Load word2vec models...')
    vocab = load_vocab(config.VOCAB_DATA)
    embeddings = get_trimmed_w2v_vectors(config.W2V_DATA)
    timer.stop()

    timer.start('Load data...')
    train = process_data(opt.train, vocab)
    if opt.val is not None:
        if opt.val != '1vs9':
            validation = process_data(opt.val, vocab)
        else:
            validation, train = train.one_vs_nine()
    else:
        validation = None

    if opt.test is not None:
        test = process_data(opt.test, vocab)
    else:
        test = None
    timer.stop()

    timer.start('Build model...')
    model = CnnModel(embeddings=embeddings)
    model.build()
    timer.stop()

    timer.start('Train model...')
    epochs = opt.e
    batch_size = opt.b
    early_stopping = True if opt.p != 0 else False
    patience = opt.p
    pre_train = opt.pre if opt.pre != '' else None
    model_name = opt.name

    model.train(
        model_name, train=train, validation=validation,
        epochs=epochs, batch_size=batch_size,
        early_stopping=early_stopping, patience=patience,
        cont=pre_train,
    )
    timer.stop()

    if test is not None:
        timer.start('Test model...')
        preds = model.predict(test, model_name)
        labels = test.labels

        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        print('Testing result:P=\t{}\tR={}\tF1={}'.format(p, r, f1))
        timer.stop()


if __name__ == '__main__':
    main()
