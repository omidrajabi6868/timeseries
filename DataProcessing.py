import pandas as pd
from tensorflow import keras
import datetime
import numpy as np
import tensorflow as tf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class DataProcessing:

    def __init__(self, csvData):
        self.csvData = csvData
        self.feature_keys = csvData.keys()[3:]
        self.split_fraction = 0.8
        self.train_split = int(self.split_fraction * int(csvData.shape[0]))
        self.step = 1
        self.past = 4*1
        self.future = 4*24
        self.batch_size = 1
        self.date_time_key = "local_timestamp"
        return

    def normalize(self, data):
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(data)
        data = imp.transform(data)
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)
        return (((data - self.data_min) / (self.data_max - self.data_min)))

    def data_cleaning(self):
        print(
            "The selected parameters are:",
            ", ".join([self.feature_keys[i] for i in [9]]),
        )
        selected_features = [self.feature_keys[i] for i in [9]]
        features = self.csvData[selected_features]
        features.index = self.csvData[self.date_time_key]
        features.head(10)

        features= self.normalize(features.values)
        features = pd.DataFrame(features)
        features.head(10)

        train_data = features.loc[0: self.train_split-1]
        val_data = features.loc[self.train_split:]

        return train_data, val_data, features

    def preparing_dataset(self):
        train_data, val_data, features = self.data_cleaning()
        start = self.past
        end = self.future + self.train_split

        x_train = train_data[[i for i in range(1)]].values
        y_train = features.iloc[start:end]

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
                if len(m) < 2:
                    decoder_inputs_train.append(np.array([-1]))
                else:
                    decoder_inputs_train.append(np.concatenate([np.array([-1]), np.squeeze(m)[:-1]], axis=0))

        x_end = len(val_data) - self.future

        label_start = self.train_split + self.past

        x_val = val_data.iloc[:x_end][[i for i in range(1)]].values
        y_val = features.iloc[label_start:]

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
                if len(m) < 2:
                    decoder_inputs_val.append(np.array([-1]))
                else:
                    decoder_inputs_val.append(np.concatenate([np.array([-1]), np.squeeze(m)[:-1]], axis=0))

        x_train = tf.squeeze((tf.convert_to_tensor(list(x_train))), axis=1)
        y_train = tf.squeeze((tf.convert_to_tensor(list(y_train))), axis=1)
        decoder_inputs_train = tf.convert_to_tensor(decoder_inputs_train)
        x_val = tf.squeeze((tf.convert_to_tensor(list(x_val))), axis=1)
        y_val = tf.squeeze((tf.convert_to_tensor(list(y_val))), axis=1)
        decoder_inputs_val = tf.convert_to_tensor(decoder_inputs_val)

        print("Input shape:", x_train.shape)
        print("Target shape:",y_train.shape)

        self.x_train = x_train
        self.y_train = y_train
        self.decoder_inputs_train = decoder_inputs_train
        self.x_val = x_val
        self.y_val = y_val
        self.decoder_inputs_val = decoder_inputs_val
        self.train_data = train_data
        self.val_data = val_data

        return x_train.shape, y_train.shape

