#!/usr/bin/env python
# encoding: utf-8
import argparse

import tensorflow as tf
import os
from tensorflow.python.keras.utils import Progbar
from models.bigru_with_attention import BiGRUAttentionModel
from models.cnn import TextCNNModel
from models.rcnn import RCNNModel
from utils.data_loader import generate_batch_data
from utils.eval import get_top_5_id, evaluate


def train_model(model):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    all_batch_count = 1
    for epoch_index in range(model.settings.max_epoch):
        train_fetches = [model.loss, model.sigmoid_y_pred, model.train_op]
        train_batch_generator = generate_batch_data('train', model.settings)
        prog = Progbar(target=model.settings.train_data_size // model.settings.batch_size)
        for index, batch in enumerate(train_batch_generator):
            feed_dict = model.create_feed_dic(batch, 0.5)
            loss, y_pred, _ = sess.run(train_fetches, feed_dict)
            if all_batch_count % 100 == 0:
                precision, recall, f1 = evaluate(batch['ground_truth'], get_top_5_id(y_pred, model.settings.batch_size))
                prog.update(index + 1, [("Loss", loss), ("precision", precision), ("recall", recall), ("F1", f1)])
            all_batch_count += 1
        test_model(model, sess)


def test_model(model, sess, is_test_mod=False):
    batch_generator = generate_batch_data('test' if is_test_mod else 'dev', model.settings)
    all_predict = []
    all_ground_truth = []
    for index, batch in enumerate(batch_generator):
        feed_dict = model.create_feed_dic(batch, 1.0)
        loss, y_pred = sess.run([model.loss, model.sigmoid_y_pred], feed_dict)
        all_predict.extend(get_top_5_id(y_pred, model.settings.batch_size))
        all_ground_truth.extend(batch['ground_truth'])
    precision, recall, f1 = evaluate(all_ground_truth, all_predict)
    print(f"\ntest result: precision: {precision} , recall: {recall} , F1: {f1}")
    if not is_test_mod:
        """说明是dev,保存模型"""
        if f1 > model.max_f1:
            model.max_f1 = f1
            save_path = model.saver.save(sess, "./checkpoints/{}/{:.3f}_{:.3f}_{:.3f}.ckpt". \
                                         format(model.model_name, precision, recall, f1))
            print("find new best model,save to path: ", save_path)


def restore_model_and_test(model):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint("./checkpoints/{}/".format(model.model_name))
    saver.restore(sess, ckpt)
    test_model(model, sess, is_test_mod=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('train_or_test', nargs='?', help='choose train or test model', choices=['train', 'test'],
                        default='train')
    parser.add_argument('--model', help="model name", choices=['bigru', 'cnn', 'rcnn'], default='bigru')
    parser.add_argument('--gpu', help="gpu device", default=4, type=int)
    ARGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
    model_map = {
        'bigru': BiGRUAttentionModel,
        'cnn': TextCNNModel,
        'rcnn': RCNNModel
    }
    if ARGS.train_or_test == 'train':
        train_model(model_map[ARGS.model]())
    else:
        restore_model_and_test(model_map[ARGS.model]())
