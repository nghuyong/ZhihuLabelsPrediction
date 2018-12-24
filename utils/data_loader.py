#!/usr/bin/env python
# encoding: utf-8
import pickle
import numpy as np
from tqdm import tqdm
from models.bigru_with_attention import BiGRUSetting


def generate_batch_data(data_type, setting):
    data_set = pickle.load(open('./data/processed/{}.pkl'.format(data_type), 'rb'))
    start_index = 0
    while True:
        if start_index + setting.batch_size > len(data_set):
            break
        one_batch_data = data_set[start_index:start_index + setting.batch_size]
        title_lengths = np.array([min(_[0], setting.title_len) for _ in one_batch_data])
        title_input = np.zeros((setting.batch_size, setting.title_len))
        detail_lengths = np.array([min(_[2], setting.detail_len) for _ in one_batch_data])
        detail_input = np.zeros((setting.batch_size, setting.detail_len))
        class_input = np.zeros((setting.batch_size, 25551))
        ground_truth = []
        for index, each in enumerate(one_batch_data):
            title = each[1]
            title = title[:min(len(each[1]), setting.title_len)]
            title_input[index, :len(title)] += np.array(title)
            detail = each[3]
            detail = detail[:min(len(each[3]), setting.title_len)]
            detail_input[index, :len(detail)] += np.array(detail)
            for topic in each[-1]:
                class_input[index, topic - 1] += 1
            ground_truth.append([topic_id - 1 for topic_id in each[-1]])
        yield {
            'title_input': title_input, 'title_lengths': title_lengths,
            'detail_input': detail_input, 'detail_lengths': detail_lengths,
            'class_input': class_input, 'ground_truth': ground_truth
        }
        start_index += setting.batch_size


if __name__ == "__main__":
    g = generate_batch_data('train', BiGRUSetting)
    for each in tqdm(g, total=BiGRUSetting.train_data_size // BiGRUSetting.batch_size):
        pass
