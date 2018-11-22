import os

import numpy as np
import tensorflow as tf
import math

from utils import Log, pad_sequences
import config

seed = 13
np.random.seed(seed)


class CnnModel:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.w2v_dim = config.W2V_DIM

        self.max_length = config.MAX_LENGTH

        self.cnn_config = config.CNN_CONFIG
        self.hidden_layers = config.HIDDEN_LAYERS

        self.all_labels = config.ALL_LABELS
        self.num_of_class = len(config.ALL_LABELS)

        self.loaded = None
        self.session = tf.Session()

    def restore_session(self, model_name):
        if self.loaded != model_name:
            saver = tf.train.Saver()
            saver.restore(self.session, 'data/trained_model/{}'.format(model_name))
            self.loaded = model_name

    def save_session(self, model_name):
        if not os.path.exists('data/trained_model/'):
            os.makedirs('data/trained_model/')

        saver = tf.train.Saver()
        saver.save(self.session, 'data/trained_model/{}'.format(model_name))

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.word_ids = tf.placeholder(name='word_ids', shape=[None, None], dtype=tf.int32)
        self.labels = tf.placeholder(name='labels', shape=[None], dtype=tf.int32)
        self.dropout_embedding = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_embedding")
        self.dropout_cnn = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_cnn')
        self.dropout_hidden = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_hidden')
        self.is_training = tf.placeholder(tf.bool, name='phase')

    def _add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        with tf.variable_scope("embedding"):
            _embeddings = tf.Variable(self.embeddings, name="lut", dtype=tf.float32, trainable=False)
            self.embeddings = tf.nn.embedding_lookup(
                _embeddings, self.word_ids,
                name="embeddings"
            )
            self.embeddings = tf.nn.dropout(self.embeddings, self.dropout_embedding)
            self.embedding_dim = self.w2v_dim

    def _add_logits_op(self):
        """
        Adds logits to self
        """

        with tf.variable_scope('cnn'):
            cnn_input = tf.expand_dims(self.embeddings, -1)
            cnn_outputs = []
            for k in self.cnn_config:
                with tf.variable_scope('cnn-{}'.format(k)):
                    filters = self.cnn_config[k]
                    height = k

                    pad_top = math.floor((k - 1) / 2)
                    pad_bottom = math.ceil((k - 1) / 2)
                    cnn_input = tf.pad(cnn_input, [[0, 0], [pad_top, pad_bottom], [0, 0], [0, 0]])

                    cnn_op = tf.layers.conv2d(
                        cnn_input, filters=filters,
                        kernel_size=(height, self.embedding_dim),
                        padding='valid', name='cnn-{}'.format(k),
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        activation=tf.nn.relu,
                    )

                    cnn_outputs.append(tf.reduce_max(cnn_op, axis=[1, 2]))
            cnn_output = tf.concat(cnn_outputs, axis=-1)
            cnn_output = tf.nn.dropout(cnn_output, self.dropout_cnn)

        with tf.variable_scope('logit'):
            output = cnn_output

            for i, v in enumerate(self.hidden_layers, start=1):
                output = tf.layers.dense(
                    inputs=output, units=v, name='hidden_{}'.format(i),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                    activation=tf.nn.tanh,
                )
                output = tf.nn.dropout(output, self.dropout_hidden)

            self.logits = tf.layers.dense(
                inputs=output, units=self.num_of_class, name='final_dense',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )

        self.pred_class = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        self.pred_prop = tf.nn.softmax(self.logits)

    def _add_loss_op(self):
        """
        Adds loss to self
        """
        with tf.variable_scope('loss_layers'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(losses)

            regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss += tf.reduce_sum(regularizer)

    def _add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            tvars = tf.trainable_variables()
            grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 100.0)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = optimizer.apply_gradients(zip(grad, tvars))

    def build(self):
        self._add_placeholders()
        self._add_word_embeddings_op()

        self._add_logits_op()
        self._add_loss_op()

        self._add_train_op()
        # f = tf.summary.FileWriter("tensorboard")
        # f.add_graph(tf.get_default_graph())
        # f.close()
        # exit(0)

    def _loss(self, feed_dict):
        feed_dict = feed_dict
        feed_dict[self.dropout_embedding] = 1.0
        feed_dict[self.dropout_cnn] = 1.0
        feed_dict[self.dropout_hidden] = 1.0
        feed_dict[self.is_training] = False

        loss = self.session.run(self.loss, feed_dict=feed_dict)

        return loss

    def _next_batch(self, dataset, batch_size):
        """

        :param dataset.Dataset dataset:
        :return:
        """
        start = 0
        while start < len(dataset.words):
            w_batch = dataset.words[start:start + batch_size]
            word_ids, _ = pad_sequences(w_batch, pad_tok=0, max_sent_length=self.max_length)

            if dataset.labels is not None:
                labels = dataset.labels[start:start + batch_size]
            else:
                labels = None

            start += batch_size
            yield {
                self.word_ids: word_ids,
                self.labels: labels
            } if labels is not None else {
                self.word_ids: word_ids
            }

    def train(self, model_name, train, validation=None, epochs=1000, batch_size=128, early_stopping=True, patience=10, verbose=True, cont=None):
        """
        :param cont:
        :param model_name:
        :param verbose:
        :param patience:
        :param early_stopping:
        :param batch_size:
        :param epochs:
        :param dataset.Dataset train:
        :param dataset.Dataset validation:
        :return:
        """
        print('Number of training examples:', len(train.labels))
        if validation is not None:
            print('Number of validation examples:', len(validation.labels))
        elif early_stopping:
            raise ValueError('Specify validation dataset to use early stopping')

        if cont is not None:
            self.restore_session(cont)
        else:
            self.session.run(tf.global_variables_initializer())

        Log.verbose = verbose

        best_loss = float('inf')
        nepoch_noimp = 0

        for e in range(epochs):
            train.shuffle_data()

            for idx, batch_data in enumerate(self._next_batch(dataset=train, batch_size=batch_size)):
                feed_dict = {
                    **batch_data,
                    self.dropout_embedding: 0.5,
                    self.dropout_cnn: 0.5,
                    self.dropout_hidden: 0.5,
                    self.is_training: True,
                }

                _, loss_train = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
                if idx % 5 == 0:
                    Log.log("Iter {}, Loss: {} ".format(idx, loss_train))

            Log.log("End epochs {}".format(e + 1))

            # stop by loss
            if early_stopping:
                total_loss = []

                for batch_data in self._next_batch(dataset=validation, batch_size=batch_size):
                    loss = self._loss(feed_dict=batch_data)
                    total_loss.append(loss)

                val_loss = np.mean(total_loss)
                Log.log('Val loss: {}'.format(val_loss))
                if val_loss < best_loss:
                    self.save_session(model_name)
                    Log.log('Save the model at epoch {}'.format(e + 1))
                    best_loss = val_loss
                    nepoch_noimp = 0
                else:
                    nepoch_noimp += 1
                    Log.log("Number of epochs with no improvement: {}".format(nepoch_noimp))
                    if nepoch_noimp >= patience:
                        Log.log('Best loss: {}'.format(best_loss))
                        break

        if not early_stopping:
            self.save_session(model_name)

    def predict(self, test, model_name, batch_size=128, pred_class=True):
        """

        :param batch_size:
        :param model_name:
        :param dataset.Dataset test:
        :return:
        """
        self.restore_session(model_name)

        y_pred = []

        for batch_data in self._next_batch(dataset=test, batch_size=batch_size):
            feed_dict = {
                **batch_data,
                self.dropout_embedding: 1.0,
                self.dropout_cnn: 1.0,
                self.dropout_hidden: 1.0,
                self.is_training: False,
            }
            if pred_class:
                preds = self.session.run(self.pred_class, feed_dict=feed_dict)
            else:
                preds = self.session.run(self.pred_prop, feed_dict=feed_dict)

            y_pred.extend(preds)

        return y_pred
