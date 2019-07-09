import time
import matplotlib.pyplot as plt
from data_process import *
from siamese_network import Siamese_network

def test():
    #测试
    predict, labels, preds = siamese_lstm.predict(train_path, model_path =
                                'model\\siames_lstm1_100.ckpt')

    TP, FP, TN, FN = 0., 0., 0., 0.
    for i in range(len(predict)):
        if predict[i] == 1 and labels[i] == 1:
            TP += 1
        elif predict[i] == 1 and labels[i] == 0:
            FP += 1
        elif predict[i] == 0 and labels[i] == 0:
            TN += 1
        else:
            FN += 1

    print(TP, FP, TN, FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print(predict[:100])
    print(labels[:100])
    F1 = 2 * precision * recall / (precision + recall)
    print('%.6f, %.6f, %.6f' % (precision, recall, F1))
    plt.hist(preds, bins=10)
    plt.show()


if __name__ == '__main__':
    #设置超参数
    start_time = time.time()
    embed_size = 128
    batch_size = 50
    learning_rate = 0.005
    epoch = 100
    vec_path = 'model\\word2vec.model'

    #数据集
    train_path = 'data\\train.csv'
    test_path = 'data\\test.csv'

    #搭建网络
    siamese_lstm = Siamese_network([embed_size, 16, 1], vec_path)
    siamese_lstm.build()

    # 训练网络
    # sents1, sents2, labels = read_data(train_path)
    # sents1, sents2, labels = np.array(sents1), np.array(sents2), np.array(labels)
    # siamese_lstm._train(learning_rate, sents1, sents2, labels, epoch, batch_size, model_path=
    #                         'model\\siames_lstm1_100.ckpt')

    #测试
    test()


















