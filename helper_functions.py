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

np.random.seed(7)

# Fills the missing values with normal distribution of mean/median and standard deviation
def fill_with_random_norm(whole_df, channel_num, period, START, meanOrmed):
    
    mean_or_med = whole_df[[whole_df.columns[channel_num]]].mean()
    if meanOrmed == "median":
        mean_or_med = whole_df[[whole_df.columns[channel_num]]].median()
    
    mu, sigma = mean_or_med, whole_df[[whole_df.columns[channel_num]]].std() # mean and standard deviation
    rand_norm = np.random.normal(mu, sigma, period)
    
    for i in range(period):
#         whole_df[whole_df.columns[channel_num]][START+i] = round(rand_norm[i], 0)
        whole_df[whole_df.columns[channel_num]][START+i] = round(rand_norm[i], 0)
        
    return whole_df
    

# Creating a dataset with splitted by time_step
def create_dataset(dataset, time_step, num_dim):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), num_dim]
        b = dataset[i+1:(i+time_step+1), num_dim]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

# Form a timeseries data based on the created_dataset
def form_timeseries(dataset, time_step, num_features):

    X_stack_of_arrs = []
    Y_stack_of_arrs = []
    for num_dim in range(0, num_features):
        
        dataX, dataY = create_dataset(dataset, time_step, num_dim)
        dataX = np.reshape(dataX, (dataX.shape[0], time_step, 1))
        dataY = np.reshape(dataY, (dataY.shape[0], time_step, 1))
        X_stack_of_arrs.append(dataX)
        Y_stack_of_arrs.append(dataY) 
    dataX = np.dstack([i for i in X_stack_of_arrs])
    dataY = np.dstack([i for i in Y_stack_of_arrs])
    
    return dataX, dataY

# Form the timeseries data back to original dataset for comparison plot
def back_to_norm_timeseries(timeseries, channel_ind):
    dataX= []
    for i in range(0, timeseries.shape[0]):
        dataX.append(timeseries[i][0][channel_ind])
    return np.array(dataX)

def back_to_norm_df(df, channel_ind):
    dataX= []
    for i in range(0, df.shape[0]):
        dataX.append(df[i][channel_ind])
    return np.array(dataX)

# Helper function to generate result
def get_result_df(predict_normalized, num_features):
    whole_result = []
    for i in range(num_features):
        
        missingPredict_channel = back_to_norm_timeseries(predict_normalized, i)
        whole_result.append(missingPredict_channel)

    return np.array(whole_result)

# Scale the oversize predicted values down to the original scale
def scale_down(df_real, filled_predicted_result, tSTART, period):
    real = np.array(df_real[df_real.columns[0]][tSTART:tSTART+period].values, dtype=float)
    pred = np.array(filled_predicted_result[tSTART:], dtype=float)

    diff = []
    new_vals = []

    for i in range(period):

        diff.append(round(pred[i]-real[i], 0))

    avg = 0

    for j in range(len(diff)):
        avg += diff[j]


    avg //= len(diff)
    print("Average: {}".format(avg))


    for i in range(period):
        new_vals.append(round(pred[i]-avg, 0))

    pred_subtracted = np.array(new_vals, dtype=float)
    print(pred_subtracted.shape)

    return pred_subtracted


