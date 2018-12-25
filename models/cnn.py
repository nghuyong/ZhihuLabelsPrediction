#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow.contrib import rnn

from models.utils import weight_variable, bias_variable


class CNNSetting:
    batch_size = 512
    title_len = 30
    detail_len = 100
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


class TextCNNModel:
    def __init__(self):
        self.model_name = 'cnn'
        self.settings = CNNSetting()
        self.n_filter_total = self.settings.n_filter * len(self.settings.filter_sizes)
        self.max_f1 = 0.0
        self.is_training = True

        with tf.name_scope('Inputs'):
            self.title_input = tf.placeholder(tf.int64, [None, self.settings.title_len], name='title_inputs')
            self.detail_input = tf.placeholder(tf.int64, [None, self.settings.detail_len], name='detail_inputs')
            self.class_input = tf.placeholder(tf.float32, [None, self.settings.class_num], name='class_input')
            self.keep_prob = tf.placeholder(tf.float32, [])

        """
        构建embedding层
        """
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding',
                                             shape=[self.settings.voc_size, self.settings.embedding_dim],
                                             initializer=tf.contrib.layers.xavier_initializer())

        """
        构建CNN层
        """
        with tf.variable_scope('cnn_text'):
            output_title = self.cnn_layer(self.title_input, self.settings.title_len)

        with tf.variable_scope('cnn_content'):
            output_content = self.cnn_layer(self.detail_input, self.settings.detail_len)

        """
        构建fully connected 层
        """
        with tf.variable_scope('fc_bn'):
            concat_output = tf.concat([output_title, output_content], axis=1)
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
            optimizer = tf.train.AdamOptimizer(learning_rate=self.settings.lr)
            self.train_op = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=1, name='cnn')

        print(f'{self.model_name} init finish')

    def create_feed_dic(self, batch_data):
        feed_dict = {self.title_input: batch_data['title_input'], self.detail_input: batch_data['detail_input'],
                     self.class_input: batch_data['class_input'],
                     self.keep_prob: 0.5}
        return feed_dict

    def cnn_layer(self, X_inputs, n_step):
        """
        TextCNN 模型。
        Args:
           X_inputs: tensor.shape=(batch_size, n_step)
        Returns:
           title_outputs: tensor.shape=(batch_size, self.n_filter_total)
        """
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        inputs = tf.expand_dims(inputs, -1)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.settings.filter_sizes):
            with tf.variable_scope("conv1%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.settings.embedding_dim, 1, self.settings.n_filter]
                W_filter = weight_variable(shape=filter_shape, name='W_filter')
                beta = bias_variable(shape=[self.settings.n_filter], name='beta_filter')
                # tf.summary.histogram('beta', beta)

                conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")

            # conv_bn, update_ema = self.batchnorm(conv, beta, convolutional=True)  # 在激活层前面加 BN
            # Apply nonlinearity, batch norm scaling is not useful with relus
            # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
            h = tf.nn.relu(conv, name="relu")

            with tf.variable_scope("conv2%s" % filter_size):
                filter_shape = [filter_size, 1, self.settings.n_filter, self.settings.n_filter]
                W_filter = weight_variable(shape=filter_shape, name='W_filter')
                beta = bias_variable(shape=[self.settings.n_filter], name='beta_filter')
                # tf.summary.histogram('beta', beta)
                conv = tf.nn.conv2d(h, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            # conv_bn, update_ema = self.batch_norm(conv, beta, convolutional=True)  # 在激活层前面加 BN
            # Apply nonlinearity, batch norm scaling is not useful with relus
            # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
            # h = tf.nn.relu(conv_bn, name="relu")
            h = tf.nn.relu(conv, name="relu")

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size * 2 + 2, 1, 1],
                                    strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_outputs.append(pooled)
            # self.update_emas.append(update_ema)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.n_filter_total])
        return h_pool_flat  # shape = [batch_size, self.n_filter_total]
