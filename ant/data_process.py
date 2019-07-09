from gensim import corpora
from gensim.models import word2vec
import os
import numpy as np
import jieba
import re
import multiprocessing
import py_compile

def read_raw_data(input_path, output_path):
    '''
    分词生成数据集
    :param input_path: 原始数据集
    :param output_path: 经过处理（去掉停用词）的数据集
    :return:
    '''
    jieba.load_userdict('data\\dict.txt')     #加入生词生成新的词典
    with open('data\\stop_words.txt', 'r', encoding = 'utf-8', errors = 'ignore') as f:
        stop_words = f.read().split('\n')

    with open(input_path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
        pattern = "[ 、*，：‘’'。！～？；【】（）( )……%￥#@,!:.;]+"

        sent_list1, sent_list2, labels = [], [], []
        for line in f:
            _, sent1, sent2, label = line.strip().split('\t')

            s1 = jieba.lcut(re.sub(pattern, '',sent1))        #将字符串中的pattern转化为空
            # sent1 = [s for s in s1 if s not in stop_words]
            sent1 = [s for s in s1]
            s2 = jieba.lcut(re.sub(pattern, '', sent2))
            # sent2 = [s for s in s2 if s not in stop_words]
            sent2 = [s for s in s2]

            if len(sent1) > 0 and len(sent2) > 0:
                sent_list1.append(sent1)
                sent_list2.append(sent2)
                labels.append(label)

    if output_path:
        with open(output_path, 'w', encoding = 'utf-8', errors = 'ignore') as f:
            for i in range(len(sent_list1)):
                sent_pair = '%s|\t|%s|\t|%s'%(' '.join(sent_list1[i]), ' '.join(sent_list2[i]), labels[i])
                f.write(sent_pair + '\n')

def read_data(data_path):
    '''
    :param data_path: 要读取的数据路径
    :return: 两个句子与标签

    '''
    sent_list1, sent_list2, labels = [], [], []
    with open(data_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            sent1, sent2, label = line.strip().split('|\t|')
            sent_list1.append(sent1.split(' '))
            sent_list2.append(sent2.split(' '))
            labels.append( [float(label)] )
    return sent_list1, sent_list2, labels


def generate_word2vec_model():
    '''
    生成词向量
    '''

    data_path = 'data\\atec_nlp_sim.csv'
    output_path = 'data\\atec.csv'
    read_raw_data(data_path, output_path)
    sent_list1, sent_list2, labels = read_data(output_path)
    sentense = sent_list1 + sent_list2
    dictionary = corpora.Dictionary(sentense)

    model = word2vec.Word2Vec(sentense, min_count = 5, size = 128, negative = 20,
                                                workers = multiprocessing.cpu_count())
    model.save('model\\word2vec.model')


def select_data(input_path, output_path, ratio = 1.5):
    '''
    原数据集中正样本与负样本不成比例（负样本太多），所以只需要一部分负样本
    :param input_path: 输入数据路径
    :param output_path: 输出数据集路径
    :param ratio: 正负样本比例
    :return:
    '''
    pos_list, neg_list = [], []
    with open(input_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            _, _, label = line.strip().split('|\t|')

            if float(label) == 1:
                pos_list.append(line)
            else:
                neg_list.append(line)

    data_list = []
    for i in np.random.choice(len(neg_list), int(len(pos_list) * ratio)):
        data_list.append(neg_list[i])
    for data in pos_list:
        data_list.append(data)

        with open(output_path, 'w', encoding = 'utf-8') as f:
            for i in range(len(data_list)):
                f.write(data_list[i])


def split_data(data_path, split_path, ratio = 0.7):
    '''
    将数据集划分为训练集与测试集
    :param data_path:
    :param split_path:
    :param ratio:
    :return:
    '''
    train = open(split_path[0], 'w', encoding = 'utf-8')
    test = open(split_path[1], 'w', encoding = 'utf-8')
    with open(data_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            if np.random.random() < ratio:
                train.write(line)
            else:
                test.write(line)

    train.close()
    test.close()


if __name__ == '__main__':
    # generate_word2vec_model()

    # model = word2vec.Word2Vec.load('model\\word2vec.model')
    # for key, item in model.similar_by_word(u'花呗', topn = 10):
    #     print('%s %s' % (key, item))
    #
    data_path = 'data\\atec.csv'
    atec_data_path = 'data\\atec_data.csv'
    split_data_path = ['data\\train.csv', 'data\\test.csv']

    # select_data(data_path, atec_data_path, ratio = 1.5)
    split_data(atec_data_path, split_data_path, ratio = 0.7)
    # py_compile.compile('data_process.py')
























