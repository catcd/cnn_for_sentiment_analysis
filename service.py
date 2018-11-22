import argparse
from sklearn.metrics import precision_recall_fscore_support
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

from utils import load_vocab, get_trimmed_w2v_vectors, Timer, parse_raw_data
from dataset import Dataset
from model import CnnModel
import config

parser = argparse.ArgumentParser(description='Service Multi-region-size CNN for Sentiment Analysis')

parser.add_argument('-pre', help='Pre-trained weight', type=str, required=True)

opt = parser.parse_args()
print('Running opt: {}'.format(opt))


def main():
    timer = Timer()
    timer.start('Load word2vec models...')
    vocab = load_vocab(config.VOCAB_DATA)
    embeddings = get_trimmed_w2v_vectors(config.W2V_DATA)
    timer.stop()

    timer.start('Build model...')
    model = CnnModel(embeddings=embeddings)
    model.build()
    model.restore_session(opt.pre)
    timer.stop()

    # Define app
    app = Flask(__name__)
    CORS(app)

    @app.route('/process', methods=['POST'])
    def process():
        data = request.get_json()
        try:
            words = [[vocab[w] if w in vocab else vocab['$UNK$'] for w in parse_raw_data(s)] for s in data['input']]
            test = Dataset(words=words)
        except:
            test = None
            abort(400)

        job_id = timer.start('Process {} example'.format(len(data['input'])))
        y_pred = model.predict(test, opt.pre, pred_class=False)

        ret = {
            'output': [i.tolist() for i in y_pred]
        }
        timer.stop(job_id)

        return jsonify(ret)

    app.run()


if __name__ == '__main__':
    main()
