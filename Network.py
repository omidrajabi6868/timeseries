import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from TransformerModel import ModelTrunk
from Time2Vec import Time2Vec
from Attention import AttentionBlock
from tensorflow.keras.utils import plot_model


class Network:

    def __init__(self, input_shape, output_shape, learning_rate= 0.001):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.n_units = 32
        return

    def simpleRNN(self):
        encoder_inputs = tf.keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = tf.keras.layers.SimpleRNN(self.n_units,
                                         return_state=True,
                                         activation="relu")
        encoder_outputs, encoder_states = encoder(encoder_inputs)

        decoder_inputs = tf.keras.layers.Input(shape=(self.output_shape[1], self.output_shape[2]))
        decoder_RNN = tf.keras.layers.SimpleRNN(self.n_units,
                                             return_sequences=True,
                                             return_state=True,
                                             activation="relu")
        decoder_outputs, _= decoder_RNN(decoder_inputs, initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(self.output_shape[2], activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = tf.keras.Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_states_inputs = tf.keras.layers.Input(shape=(self.n_units,))
        decoder_outputs, decoder_states = decoder_RNN(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = tf.keras.Model([decoder_inputs] + [decoder_states_inputs],
                                    [decoder_outputs] + [decoder_states])

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # return all models
        return model, encoder_model, decoder_model, 'SimpleRNN'

    def simpleGRU(self):
        encoder_inputs = tf.keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = tf.keras.layers.GRU(self.n_units,
                                   return_state=True,
                                   activation="relu")
        encoder_outputs, encoder_states = encoder(encoder_inputs)

        decoder_inputs = tf.keras.layers.Input(shape=(self.output_shape[1], self.output_shape[2]))
        decoder_GRU = tf.keras.layers.GRU(self.n_units,
                                       return_sequences=True,
                                       return_state=True,
                                       activation="relu")
        decoder_outputs, _ = decoder_GRU(decoder_inputs, initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(self.output_shape[2], activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = tf.keras.Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_states_inputs = tf.keras.layers.Input(shape=(self.n_units,))
        decoder_outputs, decoder_states = decoder_GRU(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = tf.keras.Model([decoder_inputs] + [decoder_states_inputs],
                                    [decoder_outputs] + [decoder_states])

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # return all models
        return model, encoder_model, decoder_model, 'GRU'

    def simpleLSTM(self):

        # Define an input sequence and process it.
        encoder_inputs = tf.keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = tf.keras.layers.LSTM(self.n_units,
                                    return_state=True,
                                    activation="relu")
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = tf.keras.layers.Input(shape=(self.output_shape[1], self.output_shape[2]))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = tf.keras.layers.LSTM(self.n_units,
                                         return_sequences=True,
                                         return_state=True,
                                         activation="relu")
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(self.output_shape[2], activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = tf.keras.Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_state_input_h = tf.keras.layers.Input(shape=(self.n_units,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(self.n_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = tf.keras.Model([decoder_inputs] + decoder_states_inputs,
                                    [decoder_outputs] + decoder_states)

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # return all models
        return model, encoder_model, decoder_model, 'LSTM'

    def bidirectional(self):
        # Define an input sequence and process it.
        encoder_inputs = tf.keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.n_units,
                                                               return_state=True,
                                                               activation="relu"))
        encoder_outputs, state_h_forward, state_c_forward, state_h_backward, state_c_backward = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h_forward, state_c_forward, state_h_backward, state_c_backward]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = tf.keras.layers.Input(shape=(self.output_shape[1], self.output_shape[2]))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_bd = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.n_units,
                                                                  return_state=True,
                                                                  return_sequences=True,
                                                                  activation="relu"))
        decoder_outputs, _, _, _, _ = decoder_bd(decoder_inputs, initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(self.output_shape[2], activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = tf.keras.Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_state_input_h_forward = tf.keras.layers.Input(shape=(self.n_units,))
        decoder_state_input_c_forward = tf.keras.layers.Input(shape=(self.n_units,))
        decoder_state_input_h_backward = tf.keras.layers.Input(shape=(self.n_units,))
        decoder_state_input_c_backward = tf.keras.layers.Input(shape=(self.n_units,))
        decoder_states_inputs = [decoder_state_input_h_forward, decoder_state_input_c_forward,
                                 decoder_state_input_h_backward, decoder_state_input_c_backward]
        decoder_outputs, state_h_forward, state_c_forward, state_h_backward, state_c_backward = decoder_bd(
            decoder_inputs, initial_state=decoder_states_inputs)

        decoder_states = [state_h_forward, state_c_forward, state_h_backward, state_c_backward]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = tf.keras.Model([decoder_inputs] + decoder_states_inputs,
                                    [decoder_outputs] + decoder_states)

        model.compile(optimizer='adam', loss='mse')
        model.summary()
        plot_model(model, to_file='model.png')

        return model, encoder_model, decoder_model, 'BD'

    def transformerModel(self):
        time2vec_dim = 1
        num_heads = 5
        head_size = 128
        ff_dim = 128
        number_layers = 5
        dropout = 0

        encoder_inputs = tf.keras.Input(shape=(self.input_shape[1], self.input_shape[2]), name="input_encoder")
        time2vec_encoder = Time2Vec(kernel_size=time2vec_dim)
        encoder_time_embedding = tf.keras.layers.TimeDistributed(time2vec_encoder, name='TimeDistributed_encoder')(encoder_inputs)
        x_encoder = K.concatenate([encoder_inputs, encoder_time_embedding], -1)
        encoder_attention = AttentionBlock(num_heads=num_heads, head_size=head_size,
                                           ff_dim=ff_dim, dropout=dropout)
        encoder_outputs = encoder_attention([x_encoder, x_encoder])

        decoder_inputs = tf.keras.layers.Input(shape=(self.output_shape[1], self.output_shape[2]), name="input_decoder")
        time2vec_decoder = Time2Vec(kernel_size=time2vec_dim)
        decoder_time_embedding = tf.keras.layers.TimeDistributed(time2vec_decoder, name='TimeDistributed_decoder')(decoder_inputs)
        x_decoder = K.concatenate([decoder_inputs, decoder_time_embedding], -1)

        decoder_attention_layers = [
            AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in
            range(number_layers)]
        for i, attention_layer in enumerate(decoder_attention_layers):
            if i is 0:
                x_decoder = attention_layer([x_decoder, x_decoder])
                # x_decoder = tf.keras.layers.Conv1D(3, 7, 2, padding='valid', activation='relu', name='decoder_conv_1')(
                #     x_decoder)
                # x_decoder = tf.keras.layers.Conv1D(3, 5, 2, padding='valid', activation='relu', name='decoder_conv_2')(
                #     x_decoder)
                # x_decoder = tf.keras.layers.Conv1D(3, 3, 2, padding='valid', activation='relu', name='decoder_conv_3')(
                #     x_decoder)
                # x_decoder = tf.keras.layers.Conv1D(3, 2, 2, padding='valid', activation='relu', name='decoder_conv_4')(
                #     x_decoder)
            else:
                x_decoder = attention_layer([encoder_outputs, x_decoder])

        x_decoder = K.reshape(x_decoder, (-1, x_decoder.shape[1] * x_decoder.shape[2]))
        decoder_dense = tf.keras.layers.Dense(self.output_shape[2], activation='relu')
        decoder_outputs = decoder_dense(x_decoder)

        model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        plot_model(model, to_file='model.png')
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        encoder_model = tf.keras.Model(encoder_inputs, encoder_outputs, name='Encoder')
        plot_model(encoder_model, to_file='Encoder.png')
        # decoder_model = tf.keras.Model(decoder_inputs, decoder_outputs, name='Decoder')
        # plot_model(decoder_model, to_file='Decoder.png')

        return model, encoder_model, decoder_outputs, 'Transformer'