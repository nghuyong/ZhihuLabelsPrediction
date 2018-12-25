#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow.contrib import rnn

from models.utils import weight_variable, bias_variable, embedding, multihead_attention, feedforward


class TransformerSetting:
    batch_size = 512
    title_len = 30
    detail_len = 100
    class_num = 25551
    voc_size = 19606
    embedding_dim = 150
    hidden_dim = 150
    fc_hidden_dim = 1024
    lr = 5e-3
    max_epoch = 50
    num_blocks = 6
    num_heads = 5
    train_data_size = 649447
    dev_data_size = 72161


class TransformerModel:
    def __init__(self):
        self.model_name = 'transformer'
        self.settings = TransformerSetting()
        self.max_f1 = 0.0
        self.is_training = True

        with tf.name_scope('Inputs'):
            self.title_input = tf.placeholder(tf.int64, [None, self.settings.title_len], name='title_inputs')
            self.detail_input = tf.placeholder(tf.int64, [None, self.settings.detail_len], name='detail_inputs')
            self.class_input = tf.placeholder(tf.float32, [None, self.settings.class_num], name='class_input')
            self.keep_prob = tf.placeholder(tf.float32, [])

        """"===========title encoder start================"""
        """
        构建embedding层
        """
        self.title_embedded, self.lookup_table = embedding(self.title_input,
                                                           vocab_size=self.settings.voc_size,
                                                           num_units=self.settings.embedding_dim,
                                                           scale=True,
                                                           scope="title_embedding")

        self.title_embedded += embedding(
            tf.tile(tf.expand_dims(tf.range(self.settings.title_len), 0), [self.settings.batch_size, 1]),
            vocab_size=self.settings.title_len,
            num_units=self.settings.embedding_dim,
            zero_pad=False,
            scale=False,
            scope="title_position_embedding")[0]

        """
        Dropout
        """
        self.title_embedded = tf.layers.dropout(self.title_embedded,
                                                rate=self.keep_prob,
                                                training=tf.convert_to_tensor(self.is_training))

        ## Blocks
        for i in range(self.settings.num_blocks):
            with tf.variable_scope("title_num_blocks_{}".format(i)):
                ### Multihead Attention
                self.title_embedded = multihead_attention(queries=self.title_embedded,
                                                          keys=self.title_embedded,
                                                          num_units=self.settings.hidden_dim,
                                                          num_heads=self.settings.num_heads,
                                                          dropout_rate=self.keep_prob,
                                                          is_training=self.is_training,
                                                          causality=False)

                ### Feed Forward
                self.title_embedded = feedforward(self.title_embedded,
                                                  num_units=[4 * self.settings.hidden_dim, self.settings.hidden_dim])

        """
        sum
        """
        self.title_encoder = tf.reduce_sum(self.title_embedded, axis=1)

        """"===========title encoder end================"""

        """"===========description encoder start================"""

        """
        构建embedding层
        """
        self.description_embedded = tf.nn.embedding_lookup(self.lookup_table, self.detail_input) * (
                    self.settings.embedding_dim ** 0.5)

        self.description_embedded += embedding(
            tf.tile(tf.expand_dims(tf.range(self.settings.detail_len), 0), [self.settings.batch_size, 1]),
            vocab_size=self.settings.detail_len,
            num_units=self.settings.embedding_dim,
            zero_pad=False,
            scale=False,
            scope="description_position_embedding")[0]

        """
        Dropout
        """
        self.description_embedded = tf.layers.dropout(self.description_embedded,
                                                      rate=self.keep_prob,
                                                      training=tf.convert_to_tensor(self.is_training))

        ## Blocks
        for i in range(self.settings.num_blocks):
            with tf.variable_scope("description_num_blocks_{}".format(i)):
                ### Multihead Attention
                self.description_embedded = multihead_attention(queries=self.description_embedded,
                                                                keys=self.description_embedded,
                                                                num_units=self.settings.hidden_dim,
                                                                num_heads=self.settings.num_heads,
                                                                dropout_rate=self.keep_prob,
                                                                is_training=self.is_training,
                                                                causality=False)

                ### Feed Forward
                self.description_embedded = feedforward(self.description_embedded,
                                                        num_units=[4 * self.settings.hidden_dim,
                                                                   self.settings.hidden_dim])

        """
        sum
        """
        self.description_encoder = tf.reduce_sum(self.description_embedded, axis=1)

        """"===========description encoder end================"""

        """
        构建fully connected层
        """
        with tf.variable_scope('fc'):
            concat_output = tf.concat([self.title_encoder, self.description_encoder], axis=1)
            W_fc = weight_variable([self.settings.hidden_dim * 2, self.settings.fc_hidden_dim], name='Weight_fc')
            fc_output = tf.matmul(concat_output, W_fc, name='h_fc')
            fc_bn_relu = tf.nn.relu(fc_output, name="relu")

        """
        构建输出层
        """
        with tf.variable_scope('output'):
            W_out = weight_variable([self.settings.fc_hidden_dim, self.settings.class_num], name='Weight_out')
            b_out = bias_variable([self.settings.class_num], name='bias_out')
            self.y_pred = tf.nn.xw_plus_b(fc_bn_relu, W_out, b_out, name='y_pred')
            self.sigmoid_y_pred = tf.nn.sigmoid(self.y_pred)

        """
        loss
        """
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_pred, labels=self.class_input)
            )

        """
        train
        """
        with tf.variable_scope('training_ops'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.settings.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=1, name='cnn')

        print(f'{self.model_name} init finish')

    def create_feed_dic(self, batch_data):
        feed_dict = {self.title_input: batch_data['title_input'], self.detail_input: batch_data['detail_input'],
                     self.class_input: batch_data['class_input'],
                     self.keep_prob: 0.1}
        return feed_dict
