import collections
import pickle
import random
import re
import thulac
import pandas as pd
import unicodedata
import tensorflow as tf
from tqdm import tqdm

url_re = re.compile(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')

data_path = 'data'
thu1 = thulac.thulac(seg_only=True, T2S=True, user_dict=data_path + '/mydict.txt')


def cleaner(text):
    text = re.sub(url_re, '<URL>', text)  # 删除URL
    text = text.strip()
    return text


def cutter(text):
    return thu1.cut(text, text=True)


class Tokenizer(object):

    def __init__(self, vocab_file):
        self.vocab = self._load_vocab(vocab_file)
        self.unk_token = "[UNK]"

    def tokenize(self, text):
        if type(text) is float:
            return []

        text = self._clean_text(text)
        text = self._clean_url(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = self._whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            token = token.lower()
            split_tokens.extend(self._run_split_on_punc(token))
        orig_tokens = self._whitespace_tokenize(" ".join(split_tokens))
        output_tokens = []
        for token in orig_tokens:
            if token not in self.vocab:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.append(token)
        return output_tokens

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    @staticmethod
    def _whitespace_tokenize(text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    @staticmethod
    def _is_punctuation(char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def convert_tokens_to_ids(self, tokens):
        return self._convert_by_vocab(self.vocab, tokens)

    @staticmethod
    def _convert_by_vocab(vocab, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    @staticmethod
    def _load_vocab(vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        with tf.gfile.GFile(vocab_file, "r") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _is_whitespace(char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    @staticmethod
    def _is_control(char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    @staticmethod
    def _clean_url(text):
        text = re.sub(url_re, 'url', text)  # 删除URL
        return text


def make_bert_pkl(from_csv_path, slice, to_pkl_path, tokenizer):
    assert len(slice) == 2
    df = pd.read_csv(from_csv_path)
    df = df.loc[slice[0] * len(df):slice[1] * len(df)]
    pkl_data = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        title_tokens = tokenizer.tokenize(row['question_title'])
        title_ids = tokenizer.convert_tokens_to_ids(title_tokens)
        detail_tokens = tokenizer.tokenize(row['question_detail'])
        detail_ids = tokenizer.convert_tokens_to_ids(detail_tokens)
        tag_ids = row['tag_ids'].split('|')

        row_output = [len(title_ids),
                      title_ids,
                      len(detail_ids),
                      detail_ids,
                      tag_ids]
        if index < 5:
            print(row_output)
        pkl_data.append(row_output)

    random.shuffle(pkl_data)
    pickle.dump(pkl_data, open(to_pkl_path, 'wb'))


if __name__ == '__main__':
    # prepare data for word based models
    # format_data('data/train_data.csv', data_path + 'data_clean.csv', cleaner)
    # format_data(data_path + '/data_clean.csv', data_path + '/data_cut.csv', cutter)
    # build_voc(data_path + '/data_cut.csv', data_path + '/voc.csv', ' ')

    # format_data('data/dev_data.csv', data_path + '/dev_data_clean.csv', cleaner)
    # format_data(data_path + '/dev_data_clean.csv', data_path + '/dev_data_cut.csv', cutter)

    # prepare data for bert (char based)
    tokenizer = Tokenizer('chinese_L-12_H-768_A-12/vocab.txt')
    make_bert_pkl('data/train_data.csv', [0, .9], 'data/processed/bert/train.pkl', tokenizer)
    make_bert_pkl('data/train_data.csv', [.9, 1], 'data/processed/bert/dev.pkl', tokenizer)
    make_bert_pkl('data/dev_data.csv', [0, 1], 'data/processed/bert/test.pkl', tokenizer)
