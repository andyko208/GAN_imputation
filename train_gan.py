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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import helper_functions as hf

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

np.random.seed(7)

# used in train - batch in train_GAN
def remove_commons(array1, array2, batch_size):
    array = []
    a = 0
    found = 0
    for i in range(array1.shape[0]):
        # added .all() to array indices because each element is another numpy array now
        if array1[i].all() == array2[0].all() and found == 0:
            found = 1
        elif found > 0 and found < batch_size:
            found+=1
        else:
            array.append(array1[i])
    return np.array(array)

def save_models(generator, discriminator, nth):
    generator.save('saved_models/generator{}.h5'.format(nth))
    discriminator.save('saved_models/discriminator{}.h5'.format(nth))

def train_gan(generator, discriminator, train, test, batch_size, save_model):
    
    val_losses = []
    train_losses = []
    generator_losses = []
    discriminator_losses = []
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True) # this works for both G and D
    
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
    @tf.function
    def generator_loss(discriminator_guess_fakes):
        return cross_entropy(tf.ones_like(discriminator_guess_fakes), discriminator_guess_fakes)
    
    @tf.function
    def discriminator_loss(discriminator_guess_reals, discriminator_guess_fakes):

        loss_fakes = cross_entropy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes)
        loss_reals = cross_entropy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)
        
        return loss_fakes + loss_reals
    
    @tf.function
    def train_step(X_batch, real_example):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

            generator_imputation = generator(X_batch)
            
            generator_imputation = tf.cast(generator_imputation, tf.float64)

        
            discriminator_guess_fakes = discriminator(generator_imputation)
            discriminator_guess_reals = discriminator(real_example)

            generator_current_loss = generator_loss(discriminator_guess_fakes)
            discriminator_current_loss = discriminator_loss(discriminator_guess_reals, discriminator_guess_fakes)
            
        generator_gradient = generator_tape.gradient(generator_current_loss, generator.trainable_variables)
        dicriminator_gradient = discriminator_tape.gradient(discriminator_current_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(dicriminator_gradient, discriminator.trainable_variables))

        return generator_current_loss, discriminator_current_loss
    
    
    
    # Always converges before 5th epoch
    num_epochs = 5
    comment_on = 0
    
    total_time = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        
        for iteration in range(train.shape[0] // batch_size):
            
            
            start = iteration * batch_size
            batch = train[start:start+batch_size]
             
            real_example = remove_commons(train, batch, batch_size)
            real_example = real_example[np.random.choice(batch_size, size=batch_size, replace=False)]
            
            # Debugging tools
            if comment_on == 1:
                print('\nbatch')
                print(batch.shape)
                print(batch)

                print('\nreal_example')
                print(real_example.shape)
                print(real_example)
            
            X_batch, Y_batch = hf.form_timeseries(batch, 4, input_dim)
            X_real_example, Y_real_example = hf.form_timeseries(real_example, 4, input_dim)
            
            # Debugging tools
            if comment_on == 1:
                print('\nX_batch')
                print(X_batch.shape)
                print(X_batch)
                print('\nY_batch')
                print(Y_batch.shape)
                print(Y_batch)
                
                print('\nX_real_example')
                print(X_real_example.shape)
                print(X_real_example)
                print('\nY_real_example')
                print(Y_real_example.shape)
                print(Y_real_example)

            
            generator_current_loss, discriminator_current_loss = train_step(X_batch, X_real_example)
            

            # Log every 50 batches
            if iteration % 50 == 0:
                
                
                
                generator_imputation = generator(X_batch)
                
                train_loss = tf.reduce_mean(tf.math.abs(generator_imputation - Y_batch))
                
#                 generator_imputation = tf.cast(generator_imputation, tf.float64)
#                 generator_imputation = tf.concat([ generator_imputation, X_batch[:,:,1:] ], axis=-1)
                
                discriminator_guess_fakes = discriminator(generator_imputation)    
                discriminator_guess_reals = discriminator(X_real_example)
                
        
#                 batch = np.random.choice(noise, batch_size, replace=False)

                batch = test[np.random.choice(batch_size, size=batch_size, replace=False)]
                X_batch, Y_batch = hf.form_timeseries(batch, 4, input_dim)


                generator_imputation = generator(X_batch, training=True)
                val_loss = tf.reduce_mean(tf.math.abs(generator_imputation - Y_batch))
                
                # plot the results after the train to find the best number of iterations
                val_losses.append(val_loss)
                train_losses.append(train_loss)
                generator_losses.append(generator_current_loss)
                discriminator_losses.append(discriminator_current_loss)

#                 print('Epoch {} - {} - {}s \nGenerator Loss: {} - Discriminator Loss: {} - Discriminator Accuracy (reals, fakes): ({}, {})'.format(
                print('Epoch {}/5 - {}/350 - {}s \nGenerator Loss: {} - Discriminator Loss: {}'.format(
                    epoch+1, iteration, round(time.time() - start_time, 1),
                    generator_current_loss,
                    discriminator_current_loss,
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.ones_like(discriminator_guess_reals), discriminator_guess_reals)),
                    tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes))
                ))
                print('Training Loss: {} - Validation Loss: {}\n'.format(train_loss, val_loss))
                start_time = time.time()
            
    print('\nTraining complete - {}s\n'.format(round(time.time() - total_time, 1)))
    
    plt.figure(figsize=(10, 7.5))
    plt.plot(val_losses, label="validation loss")
    plt.plot(train_losses, label="training loss")
    plt.plot(generator_losses, label="generator loss")
    plt.plot(discriminator_losses, label="discriminator loss")
    plt.title("Loss report visualization")
    plt.legend()
    plt.show()
    
