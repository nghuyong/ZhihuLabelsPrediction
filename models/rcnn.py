#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow.contrib import rnn

from models.utils import weight_variable, bias_variable, add_train_op


class RCNNSetting:
    batch_size = 512
    title_len = 30
    detail_len = 200
    hidden_size = 200
    n_layer = 1
    class_num = 25551
    voc_size = 19606
    embedding_dim = 150
    fc_hidden_dim = 1024
    lr = 5e-3
    max_epoch = 50
    decay_rate = 0.85
    decay_step = 15000
    train_data_size = 649447
    dev_data_size = 72161
    filter_sizes = [2, 3, 4, 5, 7]
    n_filter = 256


class RCNNModel:
    def __init__(self):
        self.model_name = 'rcnn'
        self.settings = RCNNSetting()
        self.max_f1 = 0.0
        self.n_filter_total = self.settings.n_filter * len(self.settings.filter_sizes)

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
        构建RCNN层
        """
        with tf.variable_scope('rcnn_text'):
            output_title = self.rcnn_layer(self.title_input, self.settings.title_len, self.title_length)

        with tf.variable_scope('rcnn_content'):
            output_content = self.rcnn_layer(self.detail_input, self.settings.detail_len, self.detail_length)

        concat_output = tf.concat([output_title, output_content], axis=1)

        """
        构建fully connected层
        """
        with tf.variable_scope('fc_bn'):
            W_fc = weight_variable([self.n_filter_total * 2, self.settings.fc_hidden_dim], name='Weight_fc')
            fc_output = tf.matmul(concat_output, W_fc, name='h_fc')
            fc_bn_relu = tf.nn.relu(fc_output, name="relu")
            fc_bn_drop = tf.nn.dropout(fc_bn_relu, self.keep_prob)

        """
        构建输出层
        """
        with tf.variable_scope('output'):
            W_out = weight_variable([self.settings.fc_hidden_dim, self.settings.class_num], name='Weight_out')
            b_out = bias_variable([self.settings.class_num], name='bias_out')
            self.y_pred = tf.nn.xw_plus_b(fc_bn_drop, W_out, b_out, name='y_pred')
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
            self.train_op = add_train_op(lr=self.settings.lr, loss=self.loss)

        self.saver = tf.train.Saver(max_to_keep=1, name=self.model_name)

        print(f'{self.model_name} init finish')

    def create_feed_dic(self, batch_data, keep_prob):
        feed_dict = {self.title_input: batch_data['title_input'], self.detail_input: batch_data['detail_input'],
                     self.class_input: batch_data['class_input'],
                     self.title_length: batch_data['title_lengths'], self.detail_length: batch_data['detail_lengths'],
                     self.keep_prob: keep_prob}
        return feed_dict

    def rcnn_inference(self, X_inputs, n_step, X_length):
        output_bigru = self.bi_gru(X_inputs, X_length)
        output_cnn = self.textcnn(output_bigru, n_step)
        return output_cnn  # shape = [batch_size, n_filter_total]

    def textcnn(self, X_inputs, n_step):
        """
        TextCNN 模型。
        """
        inputs = tf.expand_dims(X_inputs, -1)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.settings.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.settings.hidden_size * 2 + self.settings.embedding_dim, 1,
                                self.settings.n_filter]
                W_filter = weight_variable(shape=filter_shape, name='W_filter')
                conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(conv, name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.n_filter_total])
        return h_pool_flat  # shape = [batch_size, self.n_filter_total]

    def gru_cell(self):
        with tf.name_scope('gru_cell'):
            cell = rnn.GRUCell(self.settings.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bi_gru(self, X_inputs, X_length):
        """build the bi-GRU network. Return the encoder represented vector.
        X_inputs: [batch_size, n_step]
        n_step: 句子的词数量；或者文档的句子数。
        outputs: [fw_state, embeddings, bw_state], shape=[batch_size, hidden_size+embedding_dim+hidden_size]
        """
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)  # [batch_size, n_step, embedding_dim]
        cells_fw = [self.gru_cell() for _ in range(self.settings.n_layer)]
        cells_bw = [self.gru_cell() for _ in range(self.settings.n_layer)]
        initial_states_fw = [cell_fw.zero_state(self.settings.batch_size, tf.float32) for cell_fw in cells_fw]
        initial_states_bw = [cell_bw.zero_state(self.settings.batch_size, tf.float32) for cell_bw in cells_bw]
        outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs, sequence_length=X_length,
                                                            initial_states_fw=initial_states_fw,
                                                            initial_states_bw=initial_states_bw, dtype=tf.float32)
        hidden_outputs = tf.concat([outputs, inputs], axis=2)
        return hidden_outputs  # shape =[seg_num, n_steps, hidden_size*2+embedding_dim]

    def rcnn_layer(self, X_inputs, n_step, X_length):
        output_bigru = self.bi_gru(X_inputs, X_length)
        output_cnn = self.textcnn(output_bigru, n_step)
        return output_cnn  # shape = [batch_size, n_filter_total]
