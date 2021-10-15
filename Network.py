import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from Time2Vec import Time2Vec
from Attention import AttentionBlock
from tensorflow.keras.utils import plot_model


class Network:

    def __init__(self, input_shape, output_shape, learning_rate= 0.01):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.n_units = 128
        return

    def simpleRNN(self):
        encoder_inputs = tf.keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = tf.keras.layers.SimpleRNN(self.n_units,
                                         return_state=True,
                                         activation="relu")
        encoder_outputs, state_h = encoder(encoder_inputs)

        decoder = tf.keras.layers.SimpleRNN(self.n_units,
                                       activation='relu',
                                       return_state=True,
                                       dropout=0.
                                      )

        decoder_inp = tf.keras.layers.Input(shape=(self.output_shape[2]))

        reshapor = tf.keras.layers.Reshape((1, self.input_shape[2]))
        densor = tf.keras.layers.Dense(1, activation='relu')

        decoder_outputs = []
        decoder_input = decoder_inp
        for i in range(self.output_shape[1]):
            decoder_input = reshapor(decoder_input)
            decoder_output, state_h= decoder(decoder_input, initial_state=state_h)
            decoder_output = densor(decoder_output)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        model = tf.keras.Model([encoder_inputs, decoder_inp], decoder_outputs)

        initial_learning_rate = self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.1,
            staircase=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae'])
        model.summary()

        # return all models
        return model, 'RNN'

    def simpleGRU(self):
        # Define an input sequence and process it.
        encoder_inputs = tf.keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = tf.keras.layers.GRU(self.n_units,
                                       return_state=True,
                                       activation="relu")
        encoder_outputs, state_h = encoder(encoder_inputs)

        decoder = tf.keras.layers.GRU(self.n_units,
                                       activation='relu',
                                       return_state=True,
                                       dropout=0.,
                                       recurrent_dropout=0.,
                                       )

        decoder_inp = tf.keras.layers.Input(shape=(self.output_shape[2]))

        reshapor = tf.keras.layers.Reshape((1, self.input_shape[2]))
        densor = tf.keras.layers.Dense(1, activation='relu')

        decoder_outputs = []
        decoder_input = decoder_inp
        for i in range(self.output_shape[1]):
            decoder_input = reshapor(decoder_input)
            decoder_output, state_h= decoder(decoder_input, initial_state=state_h)
            decoder_output = densor(decoder_output)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        model = tf.keras.Model([encoder_inputs, decoder_inp], decoder_outputs)

        initial_learning_rate = self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.1,
            staircase=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae'])
        model.summary()

        # return all models
        return model, 'GRU'

    def simpleLSTM(self):

        # Define an input sequence and process it.
        encoder_inputs = tf.keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = tf.keras.layers.LSTM(self.n_units,
                                    return_state=True,
                                    activation="relu")
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        decoder = tf.keras.layers.LSTM(self.n_units,
                                       activation='relu',
                                       return_state=True,
                                       dropout=0.,
                                       recurrent_dropout=0.,
                                       )

        decoder_inp = tf.keras.layers.Input(shape=(self.output_shape[2]))

        reshapor = tf.keras.layers.Reshape((1, self.input_shape[2]))
        densor = tf.keras.layers.Dense(1, activation='relu')

        decoder_outputs = []
        decoder_input = decoder_inp
        for i in range(self.output_shape[1]):
            decoder_input = reshapor(decoder_input)
            decoder_output, state_h, state_c = decoder(decoder_input, initial_state=[state_h, state_c])
            decoder_output=densor(decoder_output)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        model = tf.keras.Model([encoder_inputs, decoder_inp], decoder_outputs)

        initial_learning_rate = self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.1,
            staircase=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae'])
        model.summary()

        # return all models
        return model, 'LSTM'

    def bidirectional(self):
        # Define an input sequence and process it.
        encoder_inputs = tf.keras.layers.Input(shape=(self.input_shape[1],self.input_shape[2]))
        encoder_layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.n_units,
                                                                             return_state=True,
                                                                             return_sequences=True,
                                                                             activation="relu"))
        encoder_outputs, h_f, c_f, h_b, c_b = encoder_layer_1(encoder_inputs)
        encoder_layer_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.n_units,
                                                                             return_state=True,
                                                                             activation="relu"))
        encoder_outputs, h_f, c_f, h_b, c_b = encoder_layer_2(encoder_outputs, initial_state=[h_f, c_f, h_b, c_b])

        decoder_layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.n_units,
                                                                     activation='relu',
                                                                     return_state=True,
                                                                     return_sequences=True,
                                                                     dropout=0.,
                                                                     recurrent_dropout=0.,
                                                                     ))

        decoder_inp = tf.keras.layers.Input(shape=(self.output_shape[2]))

        reshapor = tf.keras.layers.Reshape((1, self.input_shape[2]))
        densor = tf.keras.layers.Dense(1, activation='relu')

        decoder_outputs = []
        decoder_input = decoder_inp
        for i in range(self.output_shape[1]):
            decoder_input = reshapor(decoder_input)
            decoder_output,  h_f, c_f, h_b, c_b = decoder_layer_1(decoder_input, initial_state=[h_f, c_f, h_b, c_b])
            decoder_output = densor(decoder_output)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        model = tf.keras.Model([encoder_inputs, decoder_inp], decoder_outputs)

        initial_learning_rate = self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.1,
            staircase=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae'])
        model.summary()
        plot_model(model, to_file='model.png')

        return model, 'BD'


    def transformerModel(self):
        time2vec_dim = 1
        num_heads = 100
        ff_dim = self.n_units
        dropout = 0.1

        encoder_inputs = tf.keras.Input(shape=(self.input_shape[1], self.input_shape[2]))
        time2vec_encoder = Time2Vec(kernel_size=time2vec_dim)
        encoder_time_embedding = tf.keras.layers.TimeDistributed(time2vec_encoder)(encoder_inputs)
        x_encoder = K.concatenate([encoder_inputs, encoder_time_embedding], -1)
        encoder_attention = AttentionBlock(x_encoder.shape[2], num_heads, ff_dim, rate=dropout)
        encoder_outputs = encoder_attention([x_encoder, x_encoder], True)
        encoder_model = tf.keras.Model(encoder_inputs, encoder_outputs)
        # plot_model(encoder_model, to_file='Encoder.png')

        decoder_inputs = tf.keras.layers.Input(shape=(self.output_shape[1], self.input_shape[2]))
        time2vec_decoder = Time2Vec(kernel_size=time2vec_dim)
        decoder_time_embedding = tf.keras.layers.TimeDistributed(time2vec_decoder)(decoder_inputs)
        x_decoder = K.concatenate([decoder_inputs, decoder_time_embedding], -1)

        x_decoder = AttentionBlock(x_decoder.shape[2], num_heads, ff_dim,
                                   rate=dropout)([x_decoder, x_decoder], True, attention_mask=self.create_look_ahead_mask(x_decoder.shape[1]))
        x_decoder = AttentionBlock(x_decoder.shape[2],
                                   num_heads, ff_dim,
                                   rate=dropout)([ x_decoder, encoder_model.output],True, attention_mask=self.create_look_ahead_mask(x_decoder.shape[1]))

        x_decoder = K.reshape(x_decoder, (-1, x_decoder.shape[1] * x_decoder.shape[2]))
        decoder_dense = tf.keras.layers.Dense(self.output_shape[1], activation='relu')
        decoder_outputs = decoder_dense(x_decoder)
        # decoder_model = tf.keras.Model(decoder_inputs, decoder_outputs)
        # plot_model(decoder_model, to_file='Decoder.png')

        model = tf.keras.Model([encoder_model.inputs, decoder_inputs], decoder_outputs)
        plot_model(model, to_file='model.png')
        initial_learning_rate = self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.1,
            staircase=True
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae'])
        model.summary()

        return model, 'Transformer'

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)