#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow.contrib import rnn

from models.utils import add_train_op, add_loss, weight_variable, bias_variable, attention_layer


class BiGRUSetting:
    batch_size = 512
    title_len = 30
    detail_len = 200
    class_num = 25551
    voc_size = 19606
    embedding_dim = 150
    bi_gru_layer_num = 2
    bi_gru_hidden_dim = 200
    fc_hidden_dim = 1024
    lr = 0.001
    max_epoch = 50
    train_data_size = 649447
    dev_data_size = 72161
    test_data_size = 8946


class BiGRUAttentionModel:
    def __init__(self):
        self.model_name = 'bigru'
        self.settings = BiGRUSetting()
        self.max_f1 = 0.0

        with tf.name_scope('Inputs'):
            self.title_input = tf.placeholder(tf.int64, [None, self.settings.title_len], name='title_inputs')
            self.detail_input = tf.placeholder(tf.int64, [None, self.settings.detail_len], name='detail_inputs')
            self.class_input = tf.placeholder(tf.float32, [None, self.settings.class_num], name='class_input')
            self.title_length = tf.placeholder(tf.int64, [None], name='title_length')
            self.detail_length = tf.placeholder(tf.int64, [None], name='detail_length')
            self.keep_prob = tf.placeholder(tf.float32, [])

        """
        构建embedding层
        """
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding',
                                             shape=[self.settings.voc_size, self.settings.embedding_dim],
                                             initializer=tf.contrib.layers.xavier_initializer())

        """
        构建stack_bi_gru+Attention层
        """
        with tf.variable_scope('bi_gru_title'):
            title_embedded = tf.nn.embedding_lookup(self.embedding, self.title_input)
            title_bi_gru_output = self.stack_bi_gru_layer(title_embedded, self.title_length)
            title_attention_output = attention_layer(title_bi_gru_output, self.settings.bi_gru_hidden_dim * 2)

        with tf.variable_scope('bi_gru_detail'):
            detail_embedded = tf.nn.embedding_lookup(self.embedding, self.detail_input)
            detail_bi_gru_output = self.stack_bi_gru_layer(detail_embedded, self.detail_length)
            detail_attention_output = attention_layer(detail_bi_gru_output, self.settings.bi_gru_hidden_dim * 2)

        """
        构建fully connected层
        """
        with tf.variable_scope('fc'):
            concat_output = tf.concat([title_attention_output, detail_attention_output], axis=1)
            W_fc = weight_variable([self.settings.bi_gru_hidden_dim * 4, self.settings.fc_hidden_dim], name='Weight_fc')
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
            self.loss = add_loss(self.y_pred, self.class_input)

        """
        train
        """
        with tf.variable_scope('training_ops'):
            self.train_op = add_train_op(lr=self.settings.lr, loss=self.loss)

        self.saver = tf.train.Saver(max_to_keep=1, name=self.model_name)

        print(f'{self.model_name} init finish')

    def create_feed_dic(self, batch_data, keep_prob):
        feed_dict = {self.title_input: batch_data['title_input'], self.detail_input: batch_data['detail_input'],
                     self.class_input: batch_data['class_input'],
                     self.title_length: batch_data['title_lengths'], self.detail_length: batch_data['detail_lengths'],
                     self.keep_prob: keep_prob}
        return feed_dict

    def stack_bi_gru_layer(self, embedded_input, sequence_length):
        def gru_cell():
            with tf.name_scope('gru_cell'):
                cell = rnn.GRUCell(self.settings.bi_gru_hidden_dim, reuse=tf.get_variable_scope().reuse)
            return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        cells_fw = [gru_cell() for _ in range(self.settings.bi_gru_layer_num)]
        cells_bw = [gru_cell() for _ in range(self.settings.bi_gru_layer_num)]
        initial_states_fw = [cell_fw.zero_state(self.settings.batch_size, tf.float32) for cell_fw in cells_fw]
        initial_states_bw = [cell_bw.zero_state(self.settings.batch_size, tf.float32) for cell_bw in cells_bw]
        outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, embedded_input,
                                                            sequence_length=sequence_length,
                                                            initial_states_fw=initial_states_fw,
                                                            initial_states_bw=initial_states_bw, dtype=tf.float32)
        return outputs  # [batch_size, max_time, layers_output]
