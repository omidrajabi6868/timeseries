import pandas as pd
import numpy as np
from DataProcessing import DataProcessing
from Network import Network
import datetime
import tensorflow as tf
from DataVisualization import DataVisualization
from Prediction import predict_sequence


def main():
    train_model = False

    dataset = pd.read_csv('Data/PV.csv')
    print("The shape of raw data is ", dataset.shape)
    print(dataset.head(10))

    dv = DataVisualization(dataset)
    dv.show_raw_visualization()
    dv.show_heatmap()

    dt = DataProcessing(dataset)
    inputs_shape, targets_shape = dt.preparing_dataset()

    model, model_name = Network(input_shape=inputs_shape,
                                output_shape=targets_shape,
                                learning_rate=0.01).transformerModel()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    path_checkpoint = 'models/' + model_name + "_model_checkpoint.tf"
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=200)

    modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    encoder_inputs = dt.x_train
    decoder_outputs = dt.y_train
    if model_name=='Transformer':
        decoder_inputs_val = dt.decoder_inputs_val
        decoder_inputs_train = dt.decoder_inputs_train
    else:
        decoder_inputs_train = tf.zeros(len(dt.y_train), 1)
        decoder_inputs_val = tf.zeros(len(dt.y_val), 1)

    if train_model:
        # model.load_weights(path_checkpoint)
        history = model.fit(
            [encoder_inputs, decoder_inputs_train],
            decoder_outputs[:, :, :None],
            epochs=1000,
            batch_size=64,
            validation_data=([dt.x_val, decoder_inputs_val], dt.y_val[:, :, :None]),
            callbacks=[tensorboard_callback, es_callback, modelckpt_callback],
        )

        dv.visualize_loss(history, "Training and Validation Loss")

    model.load_weights(path_checkpoint)

    # dv.show_plot(dt.val_data, model, model_name, dt.past, dt.future, 500)

    dv.show_together(dt.val_data, inputs_shape, targets_shape, dt.past, dt.future, 400)


if __name__ == '__main__':
    main()
