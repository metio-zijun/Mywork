from keras.layers import Dense, Input
from keras.activations import softmax
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd

input = Input(batch_shape=[32, 10, 756])


HEADS_NUM = 8

def attention_heads(input):
    q = Dense(units=256)(input)
    k = Dense(units=256)(input)
    v = Dense(units=256)(input)
    k_transpose = tf.transpose(k, perm=[0, 2, 1])
    score = tf.matmul(q,k_transpose)
    score = softmax(score, axis=1)
    z = tf.matmul(score, v)
    return z

heads_lst = []
for i in range(HEADS_NUM):
    z = attention_heads(input)
    heads_lst.append(z)
z_final = tf.concat(heads_lst, axis=-1)
output = Dense(units=256)(z_final)
