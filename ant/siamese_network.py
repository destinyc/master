import tensorflow as tf
from data_process import *
import numpy as np
from gensim.models import word2vec
import time
import math
import py_compile


class Siamese_network():
    def __init__(self, layers, wv_path):
        self.ls = layers
        self.wv_model = word2vec.Word2Vec.load(wv_path)

    def get_lstm_cell(self, hidden_size):
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1.0, input_keep_prob = 1.0)

        return cell

    def bi_lstm(self, in_data, in_length):
        cell_forward = self.get_lstm_cell(self.ls[1])
        cell_backward = self.get_lstm_cell(self.ls[1])

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward,
                                    in_data, sequence_length = in_length ,dtype = tf.float32)

        return tf.concat(outputs, axis = 2)     #[batch, timestep, self.ls[1] * 2]

    def attention(self, lstm_outputs, attention_size, name):
        '''
        Attention model
        :param lstm_outputs: [ batch_size, timestep, 2*self.ls[1] ]
        :param attention_size:
        :return:attention_output(The final output of model)    [ batch_size, 2*self.ls[1] ]
        '''
        with tf.variable_scope(name) as scope:
            w = tf.Variable(tf.truncated_normal([2 * self.ls[1], attention_size]),
                                                dtype = tf.float32)
            b = tf.Variable(tf.zeros([attention_size]), dtype = tf.float32)

            w2 = tf.Variable(tf.truncated_normal([attention_size]), dtype = tf.float32)

            attention = tf.tensordot(tf.tanh(tf.tensordot(lstm_outputs, w, axes = 1) + b)
                                  , w2, axes = 1, name = 'attentionVector')

            alpha = tf.nn.softmax(attention)
            output = tf.reduce_sum(lstm_outputs * tf.expand_dims(alpha, -1), 1)  #将timestep维度消掉

            return output


    def cosine_similarity(self ,x1, x2):
        '''
        计算两个向量的余弦相似度
        :param x1:[ batch_size, self.ls[1] * 2 ]
        :param x2:[ batch_size, self.ls[1] * 2 ]
        :return:[ batch_size]
        '''
        distance = tf.reduce_sum(tf.multiply(x1, x2) ,axis = -1)

        abs_x1 = tf.sqrt(tf.reduce_sum(tf.square(x1), axis = 1))
        abs_x2 = tf.sqrt(tf.reduce_sum(tf.square(x2), axis = 1))

        similarity = tf.div(distance, tf.multiply(abs_x1, abs_x2))

        return tf.sigmoid(similarity)

    def _loss(self, y, y_hat):
        '''

        :param y: [None, self.ls[-1]]
        :param y_hat: [None, self.ls[-1]]  (网络输出是[None]， 所以要先扩展维度)
        :return:
        '''

        return tf.reduce_mean(- 1.5 * y * tf.log(tf.sigmoid(y_hat))
                              - (1 - y) * tf.log(1 - tf.sigmoid(y_hat)))

    def build(self):
        '''
        sentence[batch_size, sentencelen, wordvec]
        :return:
        '''
        self.sent1 = tf.placeholder(shape=[None, None, self.ls[0]], dtype=tf.float32)
        self.sent2 = tf.placeholder(shape=[None, None, self.ls[0]], dtype=tf.float32)
        self.sent1_len = tf.placeholder(shape=[None], dtype=tf.int32)
        self.sent2_len = tf.placeholder(shape=[None], dtype=tf.int32)

        self.y = tf.placeholder(shape=[None, self.ls[-1]], dtype = tf.float32)

        with tf.variable_scope('LSTM') as scope:
            lstm_out1 = self.bi_lstm(self.sent1, self.sent1_len)

            scope.reuse_variables()                           #将网络参数复用

            lstm_out2 = self.bi_lstm(self.sent2, self.sent1_len)

        out1 = self.attention(lstm_out1, self.ls[1] // 4, name='attention1')
        out2 = self.attention(lstm_out2, self.ls[1] // 4, name='attention2')

        sim = self.cosine_similarity(out1, out2)

        w = tf.Variable(tf.truncated_normal(shape=[1]), dtype=tf.float32)
        b = tf.Variable(tf.truncated_normal(shape=[1]), dtype=tf.float32)
        self.pred = tf.add(tf.multiply(sim, w), b)
        self.loss = self._loss(self.y, tf.expand_dims(self.pred, axis=-1))

    def get_batch(self, sent1, sent2, labels, batch_index, batch_size):
        '''

        :param sent1: 所有的句子
        :param sent2:
        :param labels:
        :param batch_index: [batch_size]   里面存储了一个batch数据对应的索引
        :param batch_size:
        :return:
        '''
        word_vocab = self.wv_model.wv.vocab

        batch_sent1 = sent1[batch_index]
        batch_sent2 = sent2[batch_index]
        batch_labels = labels[batch_index]

        max_len = 0
        sent1_len, sent2_len = [], []
        for i in range(batch_size):
            len1 = len(batch_sent1[i])
            len2 = len(batch_sent2[i])
            sent1_len.append(len1)
            sent2_len.append(len2)

            max_len = max(max(len1, len2), max_len)        #用来创建句子的向量矩阵时保持同样shape

        sent1_vec = np.zeros(shape=[batch_size, max_len, self.ls[0]])
        sent2_vec = np.zeros(shape=[batch_size, max_len, self.ls[0]])
        for i in range(batch_size):                               #获得第一个句子的向量
            for index, word in enumerate(batch_sent1[i]):
                if word in word_vocab:
                    sent1_vec[i, index] = self.wv_model[word]
        for i in range(batch_size):                                #获得第二个句子的向量
            for index, word in enumerate(batch_sent2[i]):
                if word in word_vocab:
                    sent2_vec[i, index] = self.wv_model[word]

        return sent1_vec, sent1_len, sent2_vec, sent2_len, batch_labels

    def _train(self, learning_rate, sent1, sent2, labels, epoch, batch_size, model_path = None):
        '''

        :param learning_rate:
        :param sent1: 所有句子组成的列表
        :param sent2:
        :param labels:
        :param epoch:
        :param batch_size:
        :param model_path:
        :return:
        '''

        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        saver = tf.train.Saver(max_to_keep=10)                 #只保存最近的10个模型

        n = int(len(sent1) / batch_size)
        step_start = 0
        with tf.Session() as sess:
            if model_path:
                saver.restore(sess, model_path)                 #加载已经存在的模型
                step_start = int(model_path.split('.')[0].split('_')[-1])
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            for i in range(epoch):                     #遍历epoch次数据集
                rand_index = list(range(len(sent1)))
                np.random.shuffle(rand_index)
                total_loss = 0
                for j in range(n):                      #遍历一次数据集，每次batch个
                    start = time.clock()
                    batch_index = rand_index[j * batch_size : (j + 1) * batch_size]  #选择一个batch的索引

                    sent1_vec, sent1_len, sent2_vec, sent2_len, batch_labels = self.get_batch(
                                sent1, sent2, labels, batch_index, batch_size )

                    _, loss = sess.run([train, self.loss], feed_dict =
                    {self.sent1 : sent2_vec ,self.sent2 : sent2_vec,
                     self.sent1_len : sent1_len, self.sent2_len : sent2_len,
                     self.y : batch_labels})

                    stop = time.clock()
                    print('using time: %.6f, loss: %.8f'%(stop - start, loss))

                    if math.isnan(loss):
                        f = open('log/error.txt', 'w')
                        f.write('%d\n' % j )
                        f.write('%s\n' % ' '.join(map(lambda x: str(x), batch_index)))
                        f.write('%s\n' % ' '.join(map(lambda x: str(x), sent1_len)))
                        f.write('%s\n' % ' '.join(map(lambda x: str(x), sent2_len)))

                        for bn in sent1_vec:
                            for line in bn:
                                f.write('%s\n'%(''.join([str(w) for w in line])))
                        for bn in sent2_vec:
                            for line in bn:
                                f.write('%s\n'%(''.join([str(w) for w in line])))

                        f.close()
                        exit(-1)
                    total_loss += loss

                total_loss = total_loss / n
                print('epoch: %d, loss: %s' %(i, total_loss))

                if (i + 1) % 5 == 0:
                    saver.save(sess, '.\\model\\siames_lstm1_%d.ckpt' % (i + step_start + 1))

    def predict(self, data_path, model_path = None):
        predict, _labels, preds = [], [], []
        total_loss = 0
        sents1, sents2, labels = read_data(data_path)

        with tf.Session() as sess:
            if model_path:
                saver = tf.train.Saver()
                saver.restore(sess, model_path)
            else:
                sess.run(tf.global_variables_initializer())

            for i in range(len(labels)):
                sent1, sent2, label = np.array(sents1[i]), np.array(sents2[i]), \
                                      np.array([labels[i]])

                sent1_vec, sent1_len, sent2_vec, sent2_len, batch_labels = \
                                            self.get_batch(sent1, sent2, label, [0], 1)

                pred, loss = sess.run([self.pred, self.loss],feed_dict =
                        {self.sent1 : sent1_vec, self.sent1_len : sent1_len,
                         self.sent2 : sent2_vec, self.sent2_len : sent2_len,
                         self.y : batch_labels})

                predict.append(1 if pred[0] > 0 else 0)
                preds.append(pred[0])
                total_loss += loss
                _labels.append(label[0][0])

        print('loss : %.6f' %total_loss)
        print('accuracy: %.4f'%np.mean(np.equal(predict, _labels)))
        return predict, _labels, preds

































