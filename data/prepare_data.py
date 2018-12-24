from utils.data_utils import build_voc, format_data
import re
import thulac

url_re = re.compile(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')

data_path = 'data'
thu1 = thulac.thulac(seg_only=True, T2S=True, user_dict=data_path + '/mydict.txt')


def cleaner(text):
    text = re.sub(url_re, '<URL>', text)  # 删除URL
    text = text.strip()
    return text


def cutter(text):
    return thu1.cut(text, text=True)


if __name__ == '__main__':
    # format_data('data/train_data.csv', data_path + 'data_clean.csv', cleaner)
    # format_data(data_path + '/data_clean.csv', data_path + '/data_cut.csv', cutter)
    # build_voc(data_path + '/data_cut.csv', data_path + '/voc.csv', ' ')

    format_data('data/dev_data.csv', data_path + '/dev_data_clean.csv', cleaner)
    format_data(data_path + '/dev_data_clean.csv', data_path + '/dev_data_cut.csv', cutter)
