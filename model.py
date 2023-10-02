import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas
import math
import time

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Most optimized input_dim so far
input_dim = 15
time_step = 4

kernel_size = 3
conv_filters = 32
encoder_lstm_units = 64
conv_activation = 'elu'
conv_initializer = 'he_normal'
decoder_dense_units = 64
decoder_output_activation = 'relu'
decoder_dense_initializer = 'he_normal'
padding = 'same'
batch_size = 64
learning_rate = 0.001


def build_generator():
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, RepeatVector, Conv1D, BatchNormalization,
        Concatenate, TimeDistributed, Dense
    )
    
    ## ENCODER
    # encoder_input = Input(shape=(4, 1))
    encoder_input = Input((time_step, input_dim))
    
    # LSTM block
    encoder_lstm = LSTM(units=encoder_lstm_units)(encoder_input)
    output_lstm = RepeatVector(time_step)(encoder_lstm)
    
    # Conv block
    conv_1 = Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        activation=conv_activation,
        kernel_initializer = conv_initializer,
        padding=padding)(encoder_input)

    conv_2 = Conv1D(
        filters = conv_filters,
        kernel_size = kernel_size,
        activation = conv_activation,
        kernel_initializer = conv_initializer,
        padding = padding)(conv_1)

    
    # Concatenate LSTM and Conv Encoder outputs for Decoder LSTM layer

    #     encoder_output = output_lstm
    encoder_output = Concatenate(axis = -1)([output_lstm, conv_2])

    decoder_lstm = LSTM(decoder_dense_units, return_sequences = True)(encoder_output)

    decoder_output = TimeDistributed(
        Dense(units = input_dim, 
        activation = decoder_output_activation,
        kernel_initializer = decoder_dense_initializer))(decoder_lstm)
    
              
    generator = Model(inputs = [encoder_input], outputs=[decoder_output])
    print("Generator: \n")
    generator.summary()
    
    return generator


def build_discriminator():
    '''
    Discriminator is based on the Vanilla seq2seq Encoder. The Decoder is removed
    and a Dense layer is left instead to perform binary classification.
    '''
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, RepeatVector, Conv1D, BatchNormalization,
        Concatenate, Flatten, TimeDistributed, Dense
    )

    ## ENCODER
    encoder_input = Input((None, input_dim))
    
    # LSTM block
    encoder_lstm = LSTM(units=encoder_lstm_units)(encoder_input)
    output_lstm = RepeatVector(time_step)(encoder_lstm)

    # Conv block
    conv_1 = Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        activation=conv_activation,
        kernel_initializer = conv_initializer,
        padding=padding)(encoder_input)

    conv_2 = Conv1D(
        filters = conv_filters,
        kernel_size = kernel_size,
        activation = conv_activation,
        kernel_initializer = conv_initializer,
        padding = padding)(conv_1)

    # Concatenate LSTM and Conv Encoder outputs and Flatten for Decoder LSTM layer
    encoder_output = Concatenate(axis = -1)([output_lstm, conv_2])
    encoder_output = Flatten()(encoder_output)

    # Final layer for binary classification (real/fake)
    discriminator_output = Dense(
        units = 1,
        activation = 'sigmoid',
        kernel_initializer = decoder_dense_initializer)(encoder_output)

    Discriminator = Model(inputs = [encoder_input], outputs = [discriminator_output])
    
    print("\n\nDiscriminator: \n")
    Discriminator.summary()
    
    return Discriminator

# discriminator = build_discriminator()
# discriminator.summary()

def build_GAN():
    
    generator =  build_generator()
    discriminator = build_discriminator()
    
    return generator, discriminator