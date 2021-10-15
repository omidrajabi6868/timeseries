import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from Network import Network


class DataVisualization:

    def __init__(self, df):
        self.df = df
        self.titles = df.keys()[3:]
        self.feature_keys = df.keys()[3:]
        self.colors = [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        self.date_time_key = "local_timestamp"
        return

    def show_raw_visualization(self):
        data = self.df
        time_data = data[self.date_time_key]
        fig, axes = plt.subplots(
            nrows=5, ncols=2, figsize=(15, 20),
            dpi=80, facecolor="w", edgecolor="k")
        for i in range(len(self.feature_keys)):
            key = self.feature_keys[i]
            c = self.colors[i % (len(self.colors))]
            t_data = data[key]
            t_data.index = time_data
            t_data.head()
            ax = t_data.plot(
                ax=axes[i // 2, i % 2],
                color=c,
                title="{}".format(self.titles[i]),
                rot=25)
            ax.legend([self.titles[i]])
        plt.tight_layout()
        plt.show()

    def show_heatmap(self):
        data = self.df.drop(self.df.keys()[:2], axis=1)
        plt.matshow(data.corr())
        plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
        plt.gca().xaxis.tick_bottom()
        plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=13)
        plt.title("Feature Correlation Heatmap", fontsize=14)
        plt.show()

    def visualize_loss(self, history, title):
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def show_plot(self, val_data, model, model_name, past, future, n_steps):

        predicts, true_y= self.prediction(val_data, model, model_name, past, future, n_steps)
        labels = ["True Future", "Model Prediction"]
        marker = [".-", "rx-"]

        time_steps = list(range(-(len(true_y)), 0))
        plt.title('')
        plt.plot(time_steps, true_y, marker[0], markersize=0.1, label=labels[0])
        plt.plot(time_steps, predicts, marker[1], markersize=0.1, label=labels[1])
        plt.legend()

        plt.xlabel("Time-Step")
        if future == 4:
            plt.savefig('outputs/An hour ahead_' + model_name + '.png')
        if future ==96:
            plt.savefig('outputs/a day ahead_' + model_name+ '.png')
        plt.show()

        return

    def show_together(self, val_data, inputs_shape, targets_shape, past, future, n_steps):

        transformermodel, transformermodel_name = Network(input_shape=inputs_shape,
                                     output_shape=targets_shape,
                                     learning_rate=0.01).transformerModel()

        bi_model, bi_model_name = Network(input_shape=inputs_shape,
                              output_shape=targets_shape,
                              learning_rate=0.01).bidirectional()

        lstm_model, lstm_model_name = Network(input_shape=inputs_shape,
                                output_shape=targets_shape,
                                learning_rate=0.01).simpleLSTM()

        gru_model, gru_model_name = Network(input_shape=inputs_shape,
                               output_shape=targets_shape,
                               learning_rate=0.01).simpleGRU()

        rnn_model, rnn_model_name = Network(input_shape=inputs_shape,
                               output_shape=targets_shape,
                               learning_rate=0.01).simpleRNN()

        transformermodel.load_weights('models/' + transformermodel_name + "_model_checkpoint.tf")
        bi_model.load_weights('models/' + bi_model_name + "_model_checkpoint.tf")
        lstm_model.load_weights('models/' + lstm_model_name + "_model_checkpoint.tf")
        gru_model.load_weights('models/' + gru_model_name + "_model_checkpoint.tf")
        rnn_model.load_weights('models/' + rnn_model_name + "_model_checkpoint.tf")

        predicts_tr, true_y = self.prediction(val_data, transformermodel, transformermodel_name, past, future, n_steps)
        predicts_bi, true_y = self.prediction(val_data, bi_model, bi_model_name, past, future, n_steps)
        predicts_lstm, true_y = self.prediction(val_data, lstm_model, lstm_model_name, past, future, n_steps)
        predicts_gru, true_y = self.prediction(val_data, gru_model, gru_model_name, past, future, n_steps)
        predicts_rnn, true_y = self.prediction(val_data, rnn_model, rnn_model_name, past, future, n_steps)

        labels = ["True Future", transformermodel_name, bi_model_name,
                  lstm_model_name, gru_model_name, rnn_model_name]

        marker = ["-", "--", "-.", ":", ":", ":"]

        time_steps = list(range(-(len(true_y)), 0))
        plt.figure(figsize=(25, 10))
        plt.title('')
        plt.plot(time_steps, true_y, marker[0], markersize=5, label=labels[0])
        plt.plot(time_steps, predicts_tr, marker[1], markersize=3, label=labels[1])
        plt.plot(time_steps, predicts_bi, marker[2], markersize=3, label=labels[2])
        plt.plot(time_steps, predicts_lstm, marker[3], markersize=3, label=labels[3])
        plt.plot(time_steps, predicts_gru, marker[4], markersize=3, label=labels[4])
        plt.plot(time_steps, predicts_rnn, marker[5], markersize=3, label=labels[5])
        plt.legend()

        plt.xlabel("Time-Step")
        if future == 4:
            plt.savefig('outputs/An hour ahead_together.png')
        if future == 96:
            plt.savefig('outputs/A day ahead_together.png')
        plt.show()

    def prediction(self, val_data, model, model_name, past, future, n_steps):

        x_val = keras.preprocessing.timeseries_dataset_from_array(
            val_data,
            targets=None,
            sequence_length=past,
            sampling_rate=1,
            batch_size=1,
        )

        y_val = keras.preprocessing.timeseries_dataset_from_array(
            val_data[past:],
            targets=None,
            sequence_length=future,
            sampling_rate=1,
            batch_size=1,
        )

        decoder_inputs_val = []
        for mat in y_val:
            for m in mat:
                if len(m) < 2:
                    decoder_inputs_val.append(np.array([-1]))
                else:
                    decoder_inputs_val.append(np.concatenate([np.array([-1]), np.squeeze(m)[:-1]], axis=0))

        predicts = np.array([])
        true_y = np.array([])
        i = 0
        for x, y in zip(x_val.take(n_steps), y_val.take(n_steps)):

            if model_name == 'Transformer':
                predicted_data = model.predict([x, np.expand_dims(decoder_inputs_val[i], axis=0)])
            else:
                predicted_data = model.predict([x, np.array([-1])])

            predicted_data = np.array([item for item in predicted_data])
            predicted_data = np.squeeze(predicted_data)
            if i == 0:
                predicts = np.concatenate([predicts, predicted_data], axis=0)
                true_y = np.concatenate([true_y, np.squeeze(y)], axis=0)
            predicts = np.concatenate([predicts, np.array([predicted_data[-1]])], axis=0)
            true_y = np.concatenate([true_y, np.array([np.squeeze(y)[-1]])], axis=0)

            i += 1

        return predicts, true_y





