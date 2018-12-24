import csv
import os

from tqdm import tqdm
from collections import Counter


def build_voc(data_path, output_path, split_char=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(data_path, 'rt', encoding='utf-8') as fr, open(output_path, 'wt', encoding='utf-8') as fw:
        reader = csv.reader(fr)
        rows = list(reader)
        word_list = []
        for row in tqdm(rows[1:]):
            if split_char is not None:
                word_list.extend(row[1].split(split_char))
                word_list.extend(row[2].split(split_char))
            else:
                word_list.extend(list(row[1]))
                word_list.extend(list(row[2]))
        word_list = filter(None, word_list)
        print('counting word')
        counter = Counter(word_list)
        print('building voc')
        word2id = [['word', 'word_id', 'word_frequency'],
                   ['<UNK>', 0, 0],
                   ['<PAD>', 1, 0]]
        for idx, (word, freq) in enumerate(counter.most_common()):
            word2id.append([word, idx + 2, freq])
        writer = csv.writer(fw)
        writer.writerows(word2id)
    print('build_voc finish!')


def format_data(data_path, output_path, preprocessor):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(data_path, 'rt', encoding='utf-8') as fr, open(output_path, 'wt', encoding='utf-8') as fw:
        reader = csv.reader(fr.readlines())
        writer = csv.writer(fw)
        rows = list(reader)
        for row in tqdm(rows[1:]):
            row[1] = preprocessor(row[1])
            row[2] = preprocessor(row[2])
        writer.writerows(rows)
    print('format_data finish!')
