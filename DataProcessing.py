import pandas as pd
from tensorflow import keras
import datetime
import numpy as np
import tensorflow as tf


class DataProcessing:

    def __init__(self, csvData):
        self.csvData = csvData
        self.feature_keys = csvData.keys()[3:]
        self.split_fraction = 0.7
        self.train_split = int(self.split_fraction * int(csvData.shape[0]))
        self.step = 1
        self.past = 4
        self.future = 92
        self.batch_size = 1
        self.date_time_key = "local_timestamp"
        return

    def normalize(self, data):
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)
        return ((data - self.data_min) / (self.data_max - self.data_min))

    def data_cleaning(self):
        print(
            "The selected parameters are:",
            ", ".join([self.feature_keys[i] for i in [9]]),
        )
        selected_features = [self.feature_keys[i] for i in [9]]
        features = self.csvData[selected_features]
        features.index = self.csvData[self.date_time_key]
        features.head()

        features = self.normalize(features.values)
        features = pd.DataFrame(features)
        features.head()

        train_data = features.loc[0: self.train_split - 1]
        val_data = features.loc[self.train_split:]

        return train_data, val_data, features

    def preparing_dataset(self):
        train_data, val_data, features = self.data_cleaning()
        start = self.past
        end = self.future + self.train_split

        x_train = train_data[[i for i in range(1)]].values
        y_train = features.iloc[start:end][[0]]

        sequence_length = int(self.past / self.step)
        x_train = keras.preprocessing.timeseries_dataset_from_array(
            x_train,
            targets=None,
            sequence_length=sequence_length,
            sampling_rate=self.step,
            batch_size=self.batch_size,
        )

        y_train = keras.preprocessing.timeseries_dataset_from_array(
            y_train,
            targets=None,
            sequence_length=self.future,
            sampling_rate=self.step,
            batch_size=self.batch_size,
        )

        decoder_inputs_train = []
        for mat in y_train:
            for m in mat:
                decoder_inputs_train.append(np.array([0] + list(np.squeeze(m)[:-1])))

        x_end = len(val_data) - self.future

        label_start = self.train_split + self.past

        x_val = val_data.iloc[:x_end][[i for i in range(1)]].values
        y_val = features.iloc[label_start:][[0]]

        x_val = keras.preprocessing.timeseries_dataset_from_array(
            x_val,
            targets=None,
            sequence_length=sequence_length,
            sampling_rate=self.step,
            batch_size=self.batch_size,
        )

        y_val = keras.preprocessing.timeseries_dataset_from_array(
            y_val,
            targets=None,
            sequence_length=self.future,
            sampling_rate=self.step,
            batch_size=self.batch_size,
        )

        decoder_inputs_val = []
        for mat in y_val:
            for m in mat:
                decoder_inputs_val.append(np.array([0] + list(np.squeeze(m)[:-1])))

        x_train = tf.squeeze((tf.convert_to_tensor(list(x_train))), axis=1)
        y_train = tf.squeeze((tf.convert_to_tensor(list(y_train))), axis=1)
        decoder_inputs_train = tf.convert_to_tensor(decoder_inputs_train)
        x_val = tf.squeeze((tf.convert_to_tensor(list(x_val))), axis=1)
        y_val = tf.squeeze((tf.convert_to_tensor(list(y_val))), axis=1)
        decoder_inputs_val = tf.convert_to_tensor(decoder_inputs_val)

        print("Input shape:", x_train.shape)
        print("Target shape:",y_train.shape)

        return x_train, y_train, decoder_inputs_train, x_val, y_val, decoder_inputs_val, x_train.shape, y_train.shape

