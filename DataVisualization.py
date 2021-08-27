import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


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
        data = self.df
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

    def show_plot(self, plot_data, delta, title):
        labels = ["History", "True Future", "Model Prediction"]
        marker = [".-", "rx-", "go-"]
        time_steps = list(range(-(plot_data[0].shape[0]), 0))
        if delta:
            future = np.arange(start=0, stop=delta)
        else:
            future = 0

        plt.title(title)
        for i, val in enumerate(plot_data):
            if i:
                plt.plot(future, np.array(plot_data[i]).flatten(), marker[i], label=labels[i])
            else:
                plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        # plt.xlim([time_steps[0], (future + 5) * 2])
        plt.xlabel("Time-Step")
        plt.show()
        return

    def plot_val_data(self, x_val, y_val, model):
        labels = ["True Future", "Model Prediction"]
        marker = [".-", "rx"]

        for x, y in zip(x_val.take(3), y_val.take(3)):
            x_list = []
            x_prediction = []
            for i, item in enumerate(x):
                if i == 0:
                    x_list = [m.numpy()[0] for m in item]
                    x_prediction.append(model.predict(tf.expand_dims(item, 0))[0])
                else:
                    x_list.append(item.numpy()[-1][0])
                    item = tf.concat([item[:-i], np.array(x_prediction)], axis=0)
                    x_prediction.append(model.predict(tf.expand_dims(item, 0))[0])

            x_prediction = [i[0] for i in x_prediction]
            time_steps = list(range(-(len(x_list)), 0))
            plt.figure()
            plt.title("show validation data")
            plt.plot(time_steps, x_list, marker[0], markersize=0.1, label=labels[0])
            plt.plot(list(range(-len(x_prediction), 0)), x_prediction, markersize=0.1, label=labels[1])
            plt.legend()
            plt.xlabel("Time-Step")
            plt.show()

        return


