### Adopted for benchmarking analysis from : https://github.com/jiawei6636/Bioinfor-DeepATT upon request from Referee 3.

# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layers.bidirection_rnn import BidLSTM, BidGRU
from .layers.multihead_attention import MultiHeadAttention
from .layers.category_dense import CategoryDense

class DeepAtt(keras.Model):
    def __init__(self):
        super(DeepAtt, self).__init__()
        self.conv_1 = keras.layers.Conv1D(
            filters=256,
            kernel_size=30,
            strides=1,
            padding='valid',
            activation='relu')

        self.pool_1 = keras.layers.MaxPool1D(
            pool_size=3,
            strides=3,
            padding='valid')

        self.dropout_1 = keras.layers.Dropout(0.2)

        self.bidirectional_rnn = BidLSTM(16)

        self.category_encoding = tf.eye(16)[tf.newaxis, :, :]

        self.multi_head_attention = MultiHeadAttention(16, 4)

        self.dropout_2 = keras.layers.Dropout(0.2)

        # Note: Point-wise-dense == Category Dense (weight-share).
        self.point_wise_dense_1 = keras.layers.Dense(
            units=16,
            activation='relu')

        self.point_wise_dense_2 = keras.layers.Dense(
            units=16,
            activation='relu')

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation of DeepAttention model.
        :param inputs: shape = (batch_size, length, c)
        :param training: training or not.
        :param kwargs: None
        :return: shape = (batch_size, 919)
        """
        batch_size = tf.shape(inputs)[0]

        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 1000, 4]
        # Output Tensor Shape: [batch_size, 971, 1024]
        temp = self.conv_1(inputs)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 971, 1024]
        # Output Tensor Shape: [batch_size, 64, 1024]
        temp = self.pool_1(temp)

        # Dropout Layer 1
        temp = self.dropout_1(temp, training=training)

        # Bidirectional RNN Layer 1
        # Input Tensor Shape: [batch_size, 64, 1024]
        # Output Tensor Shape: [batch_size, 64, 1024]
        temp, _ = self.bidirectional_rnn(temp, training=training, mask=mask)

        # Category Multi-head Attention Layer 1
        # Input Tensor Shape: v.shape = [batch_size, 64, 1024]
        #                     k.shape = [batch_size, 64, 1024]
        #                     q.shape = [batch_size, 919, 919]
        # Output Tensor Shape: temp.shape = [batch_size, 919, 400]
        query = tf.tile(self.category_encoding, multiples=[batch_size, 1, 1])
        temp, _ = self.multi_head_attention(query, k=temp, v=temp)

        # Dropout Layer 2
        temp = self.dropout_2(temp, training=training)

        # Category Dense Layer 1 (weight-share)
        # Input Tensor Shape: [batch_size, 919, 400]
        # Output Tensor Shape: [batch_size, 919, 100]
        temp = self.point_wise_dense_1(temp)

        # Category Dense Layer 2 (weight-share)
        # Input Tensor Shape: [batch_size, 919, 100]
        # Output Tensor Shape: [batch_size, 919, 1]
        output = self.point_wise_dense_2(temp)

        #output = tf.reshape(output, [-1, 919])

        return output

    
    
class DeepAttPlus(keras.Model):
    def __init__(self):
        super(DeepAttPlus, self).__init__()
        self.conv_1 = keras.layers.Conv1D(
            filters=256,
            kernel_size=30,
            strides=1,
            padding='valid',
            activation='relu')

        self.pool_1 = keras.layers.MaxPool1D(
            pool_size=3,
            strides=3,
            padding='valid')

        self.dropout_1 = keras.layers.Dropout(0.2)

        self.bidirectional_rnn = BidLSTM(64)

        self.category_encoding = tf.eye(919)[tf.newaxis, :, :]

        self.multi_head_attention = MultiHeadAttention(64, 4)

        self.dropout_2 = keras.layers.Dropout(0.2)

        self.point_wise_dense_1 = keras.layers.Dense(
            units=64,
            activation='relu')

        self.category_dense_1 = CategoryDense(
            units=1,
            activation='relu')

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation of DeepAttention model.
        :param inputs: shape = (batch_size, length, c)
        :param training: training or not.
        :param kwargs: None
        :return: shape = (batch_size, 919)
        """
        batch_size = tf.shape(inputs)[0]

        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 1000, 4]
        # Output Tensor Shape: [batch_size, 971, 1024]
        temp = self.conv_1(inputs)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 971, 1024]
        # Output Tensor Shape: [batch_size, 64, 1024]
        temp = self.pool_1(temp)

        # Dropout Layer 1
        temp = self.dropout_1(temp, training=training)

        # Bidirectional RNN Layer 1
        # Input Tensor Shape: [batch_size, 64, 1024]
        # Output Tensor Shape: [batch_size, 64, 1024]
        temp, _ = self.bidirectional_rnn(temp, training=training, mask=mask)

        # Category Multi-head Attention Layer 1
        # Input Tensor Shape: v.shape = [batch_size, 64, 1024]
        #                     k.shape = [batch_size, 64, 1024]
        #                     q.shape = [batch_size, 919, 919]
        # Output Tensor Shape: temp.shape = [batch_size, 919, 400]
        query = tf.tile(self.category_encoding, multiples=[batch_size, 1, 1])
        temp, _ = self.multi_head_attention(query, k=temp, v=temp)

        # Dropout Layer 2
        temp = self.dropout_2(temp, training=training)

        # Category Dense Layer 1  (weight-share)
        # Input Tensor Shape: [batch_size, 919, 400]
        # Output Tensor Shape: [batch_size, 919, 100]
        temp = self.point_wise_dense_1(temp)

        # Category Dense Layer 2 （No weight-share）
        # Input Tensor Shape: [batch_size, 919, 100]
        # Output Tensor Shape: [batch_size, 919, 1]
        output = self.category_dense_1(temp)

        output = tf.reshape(output, [-1, 919])

        return output


class DeepSEA(keras.Model):
    def __init__(self):
        super(DeepSEA, self).__init__()
        self.conv_1 = keras.layers.Conv1D(
            filters=320,
            kernel_size=8,
            strides=1,
            padding='SAME',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9))

        self.pool_1 = keras.layers.MaxPool1D(
            pool_size=3,
            strides=3,
            padding='SAME')

        self.dropout_1 = keras.layers.Dropout(0.2)

        self.conv_2 = keras.layers.Conv1D(
            filters=480,
            kernel_size=8,
            strides=1,
            padding='SAME',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9))

        self.pool_2 = keras.layers.MaxPool1D(
            pool_size=3,
            strides=3,
            padding='SAME')

        self.dropout_2 = keras.layers.Dropout(0.2)

        self.conv_3 = keras.layers.Conv1D(
            filters=960,
            kernel_size=8,
            strides=1,
            padding='SAME',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9))

        self.dropout_3 = keras.layers.Dropout(0.5)

        self.flatten = keras.layers.Flatten()

        self.dense_1 = keras.layers.Dense(
            units=64,
            activation='relu',
            activity_regularizer=tf.keras.regularizers.l1(1e-08),
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9))

        self.dense_2 = keras.layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9))

    def call(self, inputs, training=None, mask=None, **kwargs):
        """
        Forward propagation of DeepSEA model.
        :param inputs: shape = (batch_size, length, c)
        :param training: training or not.
        :param kwargs: None
        :return: shape = (batch_size, 919)
        """
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 1000, 4]
        # Output Tensor Shape: [batch_size, 1000, 320]
        temp = self.conv_1(inputs)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 1000, 320]
        # Output Tensor Shape: [batch_size, 250, 320]
        temp = self.pool_1(temp)

        # Dropout Layer 1
        temp = self.dropout_1(temp, training = training)

        # Convolution Layer 2
        # Input Tensor Shape: [batch_size, 250, 320]
        # Output Tensor Shape: [batch_size, 250, 480]
        temp = self.conv_2(temp)

        # Pooling Layer 2
        # Input Tensor Shape: [batch_size, 250, 480]
        # Output Tensor Shape: [batch_size, 63, 480]
        #temp = self.pool_2(temp)

        # Dropout Layer 2
        temp = self.dropout_2(temp, training = training)

        # Convolution Layer 3
        # Input Tensor Shape: [batch_size, 63, 480]
        # Output Tensor Shape: [batch_size, 63, 960]
        temp = self.conv_3(temp)

        # Dropout Layer 3
        temp = self.dropout_3(temp, training = training)

        # Flatten Layer 1
        # Input Tensor Shape: [batch_size, 63, 960]
        # Output Tensor Shape: [batch_size, 60480]
        temp = self.flatten(temp)

        # Fully Connection Layer 1
        # Input Tensor Shape: [batch_size, 60480]
        # Output Tensor Shape: [batch_size, 925]
        temp = self.dense_1(temp)

        # Fully Connection Layer 2
        # Input Tensor Shape: [batch_size, 925]
        # Output Tensor Shape: [batch_size, 919]
        output = self.dense_2(temp)

        return output


class DanQ(keras.Model):
    def __init__(self):
        super(DanQ, self).__init__('DanQ')
        self.conv_1 = keras.layers.Conv1D(
            filters=320,
            kernel_size=26,
            strides=1,
            padding='valid',
            activation='relu')

        self.pool_1 = keras.layers.MaxPool1D(
            pool_size=3,
            strides=3,
            padding='valid')

        self.dropout_1 = keras.layers.Dropout(0.2)

        forward_layer = keras.layers.LSTM(
            units=320,
            return_sequences=True,
            return_state=True)

        backward_layer = keras.layers.LSTM(
            units=320,
            return_sequences=True,
            return_state=True,
            go_backwards=True)

        self.bidirectional_rnn = keras.layers.Bidirectional(
            layer=forward_layer,
            backward_layer=backward_layer)

        self.dropout_2 = keras.layers.Dropout(0.5)

        self.flatten = keras.layers.Flatten()

        self.dense_1 = keras.layers.Dense(
            units=64,
            activation='relu')

        self.dense_2 = keras.layers.Dense(
            units=64,
            activation='relu')

    def call(self, inputs, training=None, mask=None, **kwargs):
        """
        Forward propagation of DeepSEA model.
        :param inputs: shape = (batch_size, length, c)
        :param training: training or not.
        :param kwargs: None
        :return: shape = (batch_size, 919)
        """
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 1000, 4]
        # Output Tensor Shape: [batch_size, 975, 320]
        temp = self.conv_1(inputs)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 975, 320]
        # Output Tensor Shape: [batch_size, 75, 320]
        temp = self.pool_1(temp)

        # Dropout Layer 1
        temp = self.dropout_1(temp, training = training)

        # Bidirectional RNN layer 1
        # Input Tensor Shape: [batch_size, 75, 320]
        # Output Tensor Shape: [batch_size, 75, 640]
        temp = self.bidirectional_rnn(temp, training = training, mask=mask)
        forward_state_output = temp[1]
        backward_state_output = temp[2]

        # Dropout Layer 2
        temp = self.dropout_2(temp[0], training = training)

        # Flatten Layer 1
        # Input Tensor Shape: [batch_size, 75, 640]
        # Output Tensor Shape: [batch_size, 48000]
        temp = self.flatten(temp)

        # Fully Connection Layer 1
        # Input Tensor Shape: [batch_size, 48000]
        # Output Tensor Shape: [batch_size, 925]
        temp = self.dense_1(temp)

        # Fully Connection Layer 2
        # Input Tensor Shape: [batch_size, 925]
        # Output Tensor Shape: [batch_size, 919]
        output = self.dense_2(temp)

        return output


class DanQ_JASPAR(keras.Model):
    def __init__(self):
        super(DanQ_JASPAR, self).__init__('DanQ_JASPAR')
        self.conv_1 = keras.layers.Conv1D(
            filters=1024,
            kernel_size=30,
            strides=1,
            padding='valid',
            activation='relu')

        self.pool_1 = keras.layers.MaxPool1D(
            pool_size=3,
            strides=3,
            padding='valid')

        self.dropout_1 = keras.layers.Dropout(0.2)

        forward_layer = keras.layers.LSTM(
            units=512,
            return_sequences=True,
            return_state=True)

        backward_layer = keras.layers.LSTM(
            units=512,
            return_sequences=True,
            return_state=True,
            go_backwards=True)

        self.bidirectional_rnn = keras.layers.Bidirectional(
            layer=forward_layer,
            backward_layer=backward_layer)

        self.dropout_2 = keras.layers.Dropout(0.5)

        self.flatten = keras.layers.Flatten()

        self.dense_1 = keras.layers.Dense(
            units=64,
            activation='relu')

        self.dense_2 = keras.layers.Dense(
            units=64,
            activation='relu')

    def build(self, input_shape):
        super(DanQ_JASPAR, self).build(input_shape)
        self.set_weights_by_JASPAR()

    def call(self, inputs, training=None, mask=None, **kwargs):
        """
        Forward propagation of DanQ-JASPAR model.
        :param inputs: shape = (batch_size, length, c)
        :param training: training or not.
        :param kwargs: None
        :return: shape = (batch_size, 919)
        """
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 1000, 4]
        # Output Tensor Shape: [batch_size, 971, 1024]
        temp = self.conv_1(inputs)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 971, 1024]
        # Output Tensor Shape: [batch_size, 64, 1024]
        temp = self.pool_1(temp)

        # Dropout Layer 1
        temp = self.dropout_1(temp, training = training)

        # Bidirectional RNN layer 1
        # Input Tensor Shape: [batch_size, 64, 1024]
        # Output Tensor Shape: [batch_size, 64, 1024]
        temp = self.bidirectional_rnn(temp, training=training, mask=mask)
        forward_state_output = temp[1]
        backward_state_output = temp[2]

        # Dropout Layer 2
        temp = self.dropout_2(temp[0], training = training)

        # Flatten Layer 1
        # Input Tensor Shape: [batch_size, 64, 1024]
        # Output Tensor Shape: [batch_size, 65536]
        temp = self.flatten(temp)

        # Fully Connection Layer 1
        # Input Tensor Shape: [batch_size, 65536]
        # Output Tensor Shape: [batch_size, 925]
        temp = self.dense_1(temp)

        # Fully Connection Layer 2
        # Input Tensor Shape: [batch_size, 925]
        # Output Tensor Shape: [batch_size, 919]
        output = self.dense_2(temp)

        return output

    def set_weights_by_JASPAR(self):
        JASPAR_motifs = np.load('./data/JASPAR_CORE_2016_vertebrates.npy', allow_pickle=True, encoding='bytes')
        JASPAR_motifs = list(JASPAR_motifs) # shape = (519, )
        reverse_motifs = [JASPAR_motifs[19][::-1, ::-1], JASPAR_motifs[97][::-1, ::-1], JASPAR_motifs[98][::-1, ::-1],
                          JASPAR_motifs[99][::-1, ::-1], JASPAR_motifs[100][::-1, ::-1], JASPAR_motifs[101][::-1, ::-1]]
        JASPAR_motifs = JASPAR_motifs + reverse_motifs # shape = (525, )

        conv_weights = self.conv_1.get_weights()
        for i in range(len(JASPAR_motifs)):
            motif = JASPAR_motifs[i][::-1, :]
            length = len(motif)
            start = np.random.randint(low=3, high=30-length+1-3)
            conv_weights[0][start:start+length, :, i] = motif - 0.25
            conv_weights[1][i] = np.random.uniform(low=-1.0, high=0.0)

        self.conv_1.set_weights(conv_weights)
