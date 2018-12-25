#!/usr/bin/env python
# encoding: utf-8
import argparse

import tensorflow as tf
import os
from tensorflow.python.keras.utils import Progbar
from models.bigru_with_attention import BiGRUAttentionModel
from models.cnn import TextCNNModel
from models.rcnn import RCNNModel
from models.transformer import TransformerModel
from models.bert_wrapper import BertWrapper
from models.bert import get_assignment_map_from_checkpoint
from utils.data_loader import generate_batch_data
from utils.eval import get_top_5_id, evaluate


def train_model(model, sess):
    model.train_writer.add_graph(sess.graph)
    for epoch_index in range(model.settings.max_epoch):
        train_fetches = [model.loss, model.sigmoid_y_pred, model.train_op, model.global_step]
        train_batch_generator = generate_batch_data('train', model.settings)
        prog = Progbar(target=model.settings.train_data_size // model.settings.batch_size)
        for index, batch in enumerate(train_batch_generator):
            feed_dict = model.create_feed_dic(batch)
            loss, y_pred, _, global_step = sess.run(train_fetches, feed_dict)
            if global_step % 100 == 0:
                precision, recall, f1 = evaluate(batch['ground_truth'], get_top_5_id(y_pred, model.settings.batch_size))
                prog.update(index + 1, [("Loss", loss), ("precision", precision), ("recall", recall), ("F1", f1)])
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="Loss", simple_value=loss),
                    tf.Summary.Value(tag="precision", simple_value=precision),
                    tf.Summary.Value(tag="recall", simple_value=recall),
                    tf.Summary.Value(tag="F1", simple_value=f1),
                ])

                model.train_writer.add_summary(summary, global_step=global_step)
        test_model(model, sess)


def test_model(model, sess, is_test_mod=False):
    batch_generator = generate_batch_data('test' if is_test_mod else 'dev', model.settings)
    all_predict = []
    all_ground_truth = []
    all_loss = 0
    batch_len = 0
    for index, batch in enumerate(batch_generator):
        feed_dict = model.create_feed_dic(batch, is_training=False)
        loss, y_pred = sess.run([model.loss, model.sigmoid_y_pred], feed_dict)
        all_predict.extend(get_top_5_id(y_pred, model.settings.batch_size))
        all_ground_truth.extend(batch['ground_truth'])
        all_loss += loss
        batch_len += 1
    all_loss = all_loss / batch_len
    precision, recall, f1 = evaluate(all_ground_truth, all_predict)
    print(f"\ntest result: precision: {precision} , recall: {recall} , F1: {f1}")
    if not is_test_mod:
        """说明是dev,保存模型"""
        if f1 > model.max_f1:
            model.max_f1 = f1
            save_path = model.saver.save(sess, "./checkpoints/{}/{:.3f}_{:.3f}_{:.3f}.ckpt". \
                                         format(model.model_name, precision, recall, f1))
            print("find new best model,save to path: ", save_path)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Loss", simple_value=all_loss),
            tf.Summary.Value(tag="precision", simple_value=precision),
            tf.Summary.Value(tag="recall", simple_value=recall),
            tf.Summary.Value(tag="F1", simple_value=f1),
        ])
        global_step = tf.train.global_step(sess, model.global_step)
        model.train_writer.add_summary(summary, global_step=global_step)


def create_sess(checkpoint_path='', model=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if model is not None and type(model) is BertWrapper:
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names
         ) = get_assignment_map_from_checkpoint(tvars, checkpoint_path)
        tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        sess.run(tf.global_variables_initializer())
    else:
        if checkpoint_path:
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(tf.global_variables_initializer())
    return sess


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('train_or_test', nargs='?', help='choose train or test model', choices=['train', 'test'],
                        default='train')
    parser.add_argument('--model', help="model name",
                        choices=['bigru', 'cnn', 'rcnn', 'transformer', 'bert'],
                        default='bigru')
    parser.add_argument('--gpu', help="gpu device", default=4, type=int)
    parser.add_argument('--checkpoint', help="pre-train model checkpoint", default='')

    ARGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
    model_map = {
        'bigru': BiGRUAttentionModel,
        'cnn': TextCNNModel,
        'rcnn': RCNNModel,
        'transformer': TransformerModel,
        'bert': BertWrapper
    }
    model = model_map[ARGS.model]()
    if ARGS.train_or_test == 'train':
        sess = create_sess(ARGS.checkpoint, model)
        train_model(model, sess)
    else:
        checkpoint_path = f"./checkpoints/{model.model_name}/"
        sess = create_sess(checkpoint_path, model)
        test_model(model, sess, is_test_mod=True)
