#!/usr/bin/env python
# coding=utf8

import math


def compute_position_weighted_precision(correct_num, predict_num):
    assert (len(correct_num) == len(predict_num))
    weighted_correct = 0.0
    weighted_predict = 0.0
    for i in range(len(correct_num)):
        weighted_correct += correct_num[i] / math.log(i + 3.0)
        weighted_predict += predict_num[i] / math.log(i + 3.0)
    return weighted_correct / weighted_predict


def compute_recall(correct_num, ground_truth_num):
    return sum(correct_num) / ground_truth_num


def evaluate(ground_truth_data, predict_data, max_tag_num=5):
    ground_truth_num = 0
    correct_num = [0.0] * max_tag_num
    predict_num = [0.0] * max_tag_num

    for ground_truth, predict in zip(ground_truth_data, predict_data):
        ground_truth_num += len(ground_truth)
        assert len(predict) == max_tag_num
        for pos, tag_id in enumerate(predict):
            if tag_id == '-1':
                continue
            predict_num[pos] += 1
            if tag_id in ground_truth:
                correct_num[pos] += 1

    precision = compute_position_weighted_precision(correct_num, predict_num)
    recall = compute_recall(correct_num, ground_truth_num)
    try:
        F1 = 2 * precision * recall / (precision + recall)
    except:
        F1 = 0

    # print("precision: {}, recall: {}, F1 {}".format(precision, recall, F1))

    return precision, recall, F1


def get_top_5_id(y_pred, batch_size):
    predict_list = []
    for i in range(batch_size):
        arr = y_pred[i]
        predict_list.append(list(arr.argsort()[-5:][::-1]))
    return predict_list


def test():
    predict = [[1, 2, 3, 4, 5],
               [4, 3, 5, 5, -1]]
    gold_truth = [[1, 2, 3, 4],
                  [4, 6]]
    print(evaluate(gold_truth, predict))


if __name__ == '__main__':
    test()
