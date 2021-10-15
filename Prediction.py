import numpy as np


# generate target given source sequence
def predict_sequence(source, model, model_name, decoder_inputs, n_steps):
    source = np.expand_dims(source, axis=0)
    # state = infenc.predict(source)
    # start of sequence input
    decoder_inp = np.expand_dims(np.array([-1]), axis=0)
    # collect predictions
    output = list()
    # output_joined = list()
    if model_name in ['SimpleRNN', 'GRU']:
        yhat, h = model.predict([source, decoder_inp])
        # store prediction
        output.append(yhat[0, :, 0])
        # update state
        state = h
    if model_name == 'LSTM':
        yhat, h, c = model.predict([source,  decoder_inp])
        # store prediction
        output.append(yhat[0, :, 0])
        # update state
        state = [h, c]

    if model_name == 'BD':
        yhat, h_forward, c_forward, h_backward, c_backward = model.predict([source,  decoder_inp])
        # store prediction
        output.append(yhat[0, :, 0])
        # update state
        state = [h_forward, c_forward, h_backward, c_backward]

    if model_name == 'Transformer':
        yhat = model.predict([source,  decoder_inp])

    # update target sequence
    target_seq = yhat
    # if t < n_steps:
    #     output_joined.append(target_seq[0, -1, 0])
    # else:
    #     output_joined += list(target_seq[0,:,0])


    return np.array(output)
