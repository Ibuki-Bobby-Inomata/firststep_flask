from sklearn.preprocessing import MinMaxScaler
import pylab
import pandas as pd
from pandas import DataFrame
import matplotlib as plt
import os
import glob

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from pprint import pprint

#kerasで多クラス分類のネットワークを構築する
#moduleのimport
import tensorflow as tf
keras = tf.keras

import numpy as np

from sklearn.metrics import f1_score
from keras.callbacks import Callback

from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

import collections

from tensorflow.keras.utils import plot_model


from keras.layers import Dropout, Embedding, LSTM, Bidirectional

import pandas as pd
import matplotlib.pyplot as plt
import os as os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score as accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn

from random import random

from tensorflow.keras.layers import Input, concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model





window_size = 10
class_num = 4
n_hidden = 100

# 入力を定義
input_acc = Input(shape=(window_size,1))
input_do = Input(shape=(window_size,2))
input_rot = Input(shape=(window_size,4))
input_acc_z = Input(shape=(window_size,2))


# 入力1から結合前まで(Conv1D)
acc = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(input_acc)
# acc = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(acc)
acc = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu')(acc)
# acc = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=3, activation='relu')(acc)

acc = Model(inputs=input_acc, outputs=acc)

# 入力2から結合前まで(Conv1D)
do = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(input_do)
# do = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(do)
do = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu')(do)
# do = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=3, activation='relu')(do)

do = Model(inputs=input_do, outputs=do)

# # 入力1から結合前まで(Conv1D)
# rot = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(input_rot)
# # rot = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(rot)
# rot = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu')(rot)
# # rot = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=3, activation='relu')(rot)

# rot = Model(inputs=input_rot, outputs=rot)

# # 入力2から結合前まで(Conv1D)
# acc_z = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(input_acc_z)
# # acc_z = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu')(acc_z)
# acc_z = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu')(acc_z)
# # acc_z = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=3, activation='relu')(acc_z)
#
# acc_z = Model(inputs=input_acc_z, outputs=acc_z)

# 結合
combined = concatenate([acc.output, do.output
                           # , rot.output, acc_z.output
                        ])
# combined = tf.keras.layers.GlobalMaxPooling1D()(combined)

# # 密結合
z = tf.keras.layers.LSTM(units=n_hidden, return_sequences=True)(combined)

# z = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_hidden, return_sequences=True))(z)
# z = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_hidden, return_sequences=True))(z)
# z = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_hidden, return_sequences=True))(z)

z = tf.keras.layers.LSTM(units=n_hidden, return_sequences=False)(z)


# z = tf.keras.layers.Dense(units=2500, activation='relu')(z)
# z = tf.keras.layers.Dense(units=50, activation='relu')(z)
z = tf.keras.layers.Dense(units=10, activation='relu')(z)
z = tf.keras.layers.Dense(class_num, activation='softmax')(z)

# モデル定義とコンパイル
rnn = tf.keras.Model(inputs=[acc.input, do.input
    # , rot.input, acc_z.input
                             ], outputs=z)
rnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
rnn.summary()


# import pydotprint

plot_model(
    rnn,
    show_shapes=True,
    to_file='model.png'
)