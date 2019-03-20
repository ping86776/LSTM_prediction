#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler

lstm_hidden_size = 10   # 隐层神经元的个数
num_layers = 2  # LSTM层数
inputsize = 2   # 输入特征数
outputsize = 1   # 输出特征数
training_steps = 1  # 训练轮数
batch_size = 80  # batch大小
training_exa = 5800   # 训练数据个数
# time_step = int((training_exa ) / batch_size)        # 循环神经网络训练序列长度
time_step = 20
# ——————————————————导入数据——————————————————————
# f = open('./dataset_2.csv')
f = open('./3.csv')
df = pd.read_csv(f)     # 读入数据
data = df.iloc[:, 0:3].values  # 取第3-10列
# print(np.shape(data))
train_data = data[0:training_exa]
print(len(train_data))
test_data = data[training_exa + 1:6800]
print(len(test_data))


# 获取训练集
def get_train_data(data):
    normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)  # 标准化
    print(np.shape(normalized_data))
    train_x = []
    train_y = []
    batch_index = []
    # 用data前面的time_step个点的信息，预测第i+time_step个点的函数值
    for i in range(len(normalized_data) - time_step):
        if i % batch_size == 0:
           batch_index.append(i)
        X = normalized_data[i:i + time_step, :inputsize]
        y = normalized_data[i:i + time_step, inputsize, np.newaxis]
        train_x.append(X.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_data) - time_step))
    return batch_index, train_x, train_y
# batch_index, train_x, train_y = get_train_data(train_data)
# print(np.shape(train_x))
# print(np.shape(train_y))

# 获取测试集
def get_test_data(data):
    test_x = []
    test_y = []
    mean = np.mean(data, axis=0)   # 均值
    std = np.std(data, axis=0)  # 均方差
    normalized_data = (data - mean) / std  # 标准化
    print(std)
    size = (len(normalized_data) + time_step - 1) // time_step  # 有size个sample
    # print("size =", size)
    # 用data前面的time_step个点的信息，预测第i+time_step个点的函数值
    for i in range(size - 1):
        X = normalized_data[i * time_step:(i + 1) * time_step, :inputsize]
        y = normalized_data[i * time_step:(i + 1) * time_step, inputsize]
        test_x.append(X.tolist())
        test_y.extend(y)
    test_x.append((normalized_data[(i + 1) * time_step:, :inputsize]).tolist())
    test_y.extend((normalized_data[(i + 1) * time_step:, inputsize]).tolist())
    return mean, std, test_x, test_y

mean, std, test_x, test_y = get_test_data(test_data)
# print(np.shape(test_x))
# print(np.shape(test_y))
# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置、dropout参数

weights = {
         'in': tf.Variable(tf.random_normal([inputsize, lstm_hidden_size])),
         'out': tf.Variable(tf.random_normal([lstm_hidden_size, 1]))
        }
biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[lstm_hidden_size, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
       }
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# ——————————————————定义神经网络变量——————————————————
def lstmCell():
    # basicLstm单元
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm


def lstm(X):

    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, inputsize])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, lstm_hidden_size])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for _ in range(num_layers)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, lstm_hidden_size])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ————————————————训练模型————————————————————
def train_lstm():
    X = tf.placeholder(tf.float32, shape=[None, time_step, inputsize])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, outputsize])
    batch_index, train_x, train_y = get_train_data(train_data)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    # loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1])-tf.reshape(Y, [-1])))
    loss = tf.losses.mean_squared_error(labels=tf.reshape(Y, [-1]), predictions=tf.reshape(pred, [-1]))
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.5, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=0.0008).minimize(loss, global_step=global_step)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(training_steps):
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss],
                                    feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                               Y: train_y[batch_index[step]:batch_index[step + 1]], keep_prob: 0.5})
            print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ",saver.save(sess, 'model_save2\\modle.ckpt'))
        print("The train has finished")
train_lstm()


# ————————————————预测模型————————————————————
def prediction():
    X = tf.placeholder(tf.float32, shape=[None, time_step, inputsize])
    mean, std, test_x, test_y = get_test_data(test_data)
    print(test_y)
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred,_ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X:[test_x[step]], keep_prob: 0.5})
            # print(pred)
            predict = prob.reshape((-1))
            # print(np.shape(predict))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[inputsize] + mean[inputsize]
        test_predict = np.array(test_predict) * std[inputsize] + mean[inputsize]
        print("实际值", test_y)
        print("预测值", test_predict)
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差率
        # rmse = np.sqrt(mean_squared_error(test_predict,test_y))
        print("The accuracy of this predict:", acc)
        # print("The rmse:", rmse)
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, label='predictions',)
        plt.plot(list(range(len(test_y))), test_y, label='reals')
        plt.xlabel("testing_num")
        plt.ylabel("scheduler")
        plt.legend()
        plt.show()

prediction()
