import numpy as np


# generate target given source sequence
def predict_sequence(infenc, infdec, model_name, source, n_steps, y):

    source = np.expand_dims(source, axis=0)
    state = infenc.predict(source)
    # start of sequence input
    target_seq = np.expand_dims(y, axis=0)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        if model_name in ['SimpleRNN', 'GRU']:
            yhat, h= infdec.predict([target_seq] + [state])
            # store prediction
            output.append(yhat[0, :, 0])
            # update state
            state = h
        if model_name == 'LSTM':
            yhat, h, c = infdec.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0, :, 0])
            # update state
            state = [h, c]

        if model_name == 'BD':
            yhat, h_forward, c_forward, h_backward, c_backward = infdec.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0, :, 0])
            # update state
            state = [h_forward, c_forward, h_backward, c_backward]

        if model_name == 'Transformer':
            yhat = infdec.predict(target_seq)

        # update target sequence
        target_seq = yhat

    return np.array(output)

