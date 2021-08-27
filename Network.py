import numpy as np
import tensorflow as ts
from tensorflow import keras


class Network:

    def __init__(self, input_shape, output_shape, learning_rate= 0.001):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.n_units = 32
        return

    def simpleRNN(self):
        encoder_inputs = keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = keras.layers.SimpleRNN(self.n_units,
                                         return_state=True,
                                         activation="relu")
        encoder_outputs, encoder_states = encoder(encoder_inputs)

        decoder_inputs = keras.layers.Input(shape=(self.output_shape[1], self.output_shape[2]))
        decoder_RNN = keras.layers.SimpleRNN(self.n_units,
                                             return_sequences=True,
                                             return_state=True,
                                             activation="relu")
        decoder_outputs, _= decoder_RNN(decoder_inputs, initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(self.output_shape[2], activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = keras.Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_states_inputs = keras.layers.Input(shape=(self.n_units,))
        decoder_outputs, decoder_states = decoder_RNN(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model([decoder_inputs] + [decoder_states_inputs],
                                    [decoder_outputs] + [decoder_states])

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # return all models
        return model, encoder_model, decoder_model, 'SimpleRNN'

    def simpleGRU(self):
        encoder_inputs = keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = keras.layers.GRU(self.n_units,
                                   return_state=True,
                                   activation="relu")
        encoder_outputs, encoder_states = encoder(encoder_inputs)

        decoder_inputs = keras.layers.Input(shape=(self.output_shape[1], self.output_shape[2]))
        decoder_GRU = keras.layers.GRU(self.n_units,
                                       return_sequences=True,
                                       return_state=True,
                                       activation="relu")
        decoder_outputs, _ = decoder_GRU(decoder_inputs, initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(self.output_shape[2], activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = keras.Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_states_inputs = keras.layers.Input(shape=(self.n_units,))
        decoder_outputs, decoder_states = decoder_GRU(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model([decoder_inputs] + [decoder_states_inputs],
                                    [decoder_outputs] + [decoder_states])

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # return all models
        return model, encoder_model, decoder_model, 'GRU'

    def simpleLSTM(self):

        # Define an input sequence and process it.
        encoder_inputs = keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = keras.layers.LSTM(self.n_units,
                                    return_state=True,
                                    activation="relu")
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.layers.Input(shape=(self.output_shape[1], self.output_shape[2]))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = keras.layers.LSTM(self.n_units,
                                         return_sequences=True,
                                         return_state=True,
                                         activation="relu")
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(self.output_shape[2], activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = keras.Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_state_input_h = keras.layers.Input(shape=(self.n_units,))
        decoder_state_input_c = keras.layers.Input(shape=(self.n_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs,
                                    [decoder_outputs] + decoder_states)

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # return all models
        return model, encoder_model, decoder_model, 'LSTM'

    def bidirectional(self):
        # Define an input sequence and process it.
        encoder_inputs = keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        encoder = keras.layers.Bidirectional(keras.layers.LSTM(self.n_units,
                                                               return_state=True,
                                                               activation="relu"))
        encoder_outputs, state_h_forward, state_c_forward, state_h_backward, state_c_backward = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h_forward, state_c_forward, state_h_backward, state_c_backward]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.layers.Input(shape=(self.output_shape[1], self.output_shape[2]))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_bd = keras.layers.Bidirectional(keras.layers.LSTM(self.n_units,
                                                                  return_state=True,
                                                                  return_sequences=True,
                                                                  activation="relu"))
        decoder_outputs, _, _, _, _ = decoder_bd(decoder_inputs, initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(self.output_shape[2], activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # define inference encoder
        encoder_model = keras.Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_state_input_h_forward = keras.layers.Input(shape=(self.n_units,))
        decoder_state_input_c_forward = keras.layers.Input(shape=(self.n_units,))
        decoder_state_input_h_backward = keras.layers.Input(shape=(self.n_units,))
        decoder_state_input_c_backward = keras.layers.Input(shape=(self.n_units,))
        decoder_states_inputs = [decoder_state_input_h_forward, decoder_state_input_c_forward,
                                 decoder_state_input_h_backward, decoder_state_input_c_backward]
        decoder_outputs, state_h_forward, state_c_forward, state_h_backward, state_c_backward = decoder_bd(
            decoder_inputs, initial_state=decoder_states_inputs)

        decoder_states = [state_h_forward, state_c_forward, state_h_backward, state_c_backward]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs,
                                    [decoder_outputs] + decoder_states)

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        return model, encoder_model, decoder_model, 'BD'



    # def transformerModel(self):