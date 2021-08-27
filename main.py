import pandas as pd
import numpy as np
from DataProcessing import DataProcessing
from Network import Network
import datetime
import tensorflow as tf
from DataVisualization import DataVisualization
from Prediction import predict_sequence


def main():

    train_model = True

    dataset = pd.read_csv('PV.csv')
    print("The shape of raw data is ", dataset.shape)
    print(dataset.head(10))

    dv = DataVisualization(dataset)
    dv.show_raw_visualization()
    dv.show_heatmap()

    dt = DataProcessing(dataset)
    x_train, y_train, decoder_inputs_train, x_val, y_val, decoder_inputs_val, inputs_shape, targets_shape = dt.preparing_dataset()

    model, encoder_model, decoder_model, model_name = Network(input_shape=inputs_shape,
                                                              output_shape= targets_shape,
                                                              learning_rate=0.001).simpleLSTM()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    path_checkpoint = model_name + "_model_checkpoint.h5"
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)

    modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    encoder_inputs = x_train
    decoder_outputs = y_train

    if train_model:
        history = model.fit(
            [encoder_inputs, decoder_inputs_train],
            decoder_outputs,
            epochs=1000,
            batch_size=32,
            validation_data=([x_val, decoder_inputs_val], y_val),
            callbacks=[tensorboard_callback, es_callback, modelckpt_callback],
        )

        dv.visualize_loss(history, "Training and Validation Loss")

    model.load_weights(path_checkpoint)

    i=0
    start = 2000
    steps = 3
    for x, y in zip(x_val[start:start+steps], decoder_inputs_val[start:start+steps]):
        dv.show_plot(
            [x.numpy(), y_val[start+i:start+5+i], predict_sequence(encoder_model, decoder_model, model_name, x.numpy(), 5, y)],
            y.shape[0]*5,
            "Prediction",
        )
        i += 1

    # dv.plot_val_data(x_val, y_val, model)


if __name__ == '__main__':
    main()









