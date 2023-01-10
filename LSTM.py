print('hello, world!')
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:54:20 2020

@author: 小胡图
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import SimpleRNN, LSTM, GRU
from keras.optimizers import SGD, Nadam, Adam, RMSprop
from keras.callbacks import TensorBoard
from keras.utils import np_utils
import scipy.io 
import numpy as np

data = scipy.io.loadmat('psd1.mat')
y = scipy.io.loadmat('label.mat')

"""
将训练数据调整为LSTM的正确输入尺寸
"""
x_train=[]
x_1 = data['psd']
x_train.append(x_1)
x_2 = data['psd1_2']
x_train.append(x_2)
x_3 = data['psd1_3']
x_train.append(x_3)
x_4= data['psd1_4']
x_train.append(x_4)
x_5 = data['psd1_5']
x_train.append(x_5)
x_6 = data['psd1_6']
x_train.append(x_6)
x_7 = data['psd1_7']
x_train.append(x_7)
x_8 = data['psd1_8']
x_train.append(x_8)
x_9 = data['psd1_9']
x_train.append(x_9)
x_train=np.array(x_train).reshape((9,15,3))

x_test = []
x_10 = data['psd1_10']
x_test.append(x_10)
x_11 = data['psd1_11']
x_test.append(x_11)
x_12 = data['psd1_12']
x_test.append(x_12)
x_13 = data['psd1_13']
x_test.append(x_13)
x_14 = data['psd1_14']
x_test.append(x_14)
x_15 = data['psd1_15']
x_test.append(x_15)
x_test=np.array(x_test).reshape((6,15,3))

"""

"""
y = y['label'].reshape(15,1)
y_train=y[[0,1,2,3,4,5,6,7,8]]
y_train = np.array(y_train)
tmp_train = []
for key in y_train:
    if key== 1:
        tmp_train.append(1)
    elif key==0:
        tmp_train.append(0)
    elif key==-1:
        tmp_train.append(2)
y_train = np.array(tmp_train)
y_train = np_utils.to_categorical(y_train, 3)
y_test=y[[9,10,11,12,13,14]]
tmp_test = []
for i in y_test:
    if i == 1:
        tmp_test.append(1)
    elif i==0:
        tmp_test.append(0)
    elif i==-1:
        tmp_test.append(2)
y_test = np.array(tmp_test)
y_test = np_utils.to_categorical(y_test, 3)
"""
建立模型
"""
model = Sequential()
model.add(LSTM(10, return_sequences = True, input_shape=(15, 3)))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(5))
model.add(Dense(3, activation = 'softmax'))
model.summary()
"""
优化器设置
学习率为0.001
"""
optim = Nadam(lr = 0.001)
# 设置损失函数为交叉熵损失函数
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
"""
epochs设置为15
batch_size设置为9
"""
model.fit(x_train, y_train, epochs=100, batch_size=9)  

score, acc = model.evaluate(x_test, y_test,
                            batch_size=1)
print('测试得分:', score)
print('测试精度:', acc)