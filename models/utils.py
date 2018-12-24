#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf


def weight_variable(shape, name):
    """创建一个参数W"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """创建一个参数b"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def attention_layer(input_sequence, output_size):
    # 输入 [batch_size, max_time, layers_output]
    initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope('attention'):
        attention_context_vector = tf.get_variable(name='attention_context_vector', shape=[output_size],
                                                   initializer=initializer, dtype=tf.float32)
        input_projection = tf.contrib.layers.fully_connected(input_sequence, output_size, activation_fn=tf.tanh)
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = tf.multiply(input_sequence, attention_weights)
        outputs = tf.reduce_sum(weighted_projection, axis=1)
        # 输出 [batch_size, hidden_size*2]
        return outputs


def add_loss(y_pred, y_label):
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_label)
    )
    return loss


def add_train_op(lr, loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)
    return train_op
