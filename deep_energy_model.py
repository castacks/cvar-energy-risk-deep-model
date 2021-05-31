import numpy as np
from tcn import TCN
import tensorflow as tf
import os

def load_inputs(lookback=20,
                batch_size=32,
                nb_filters=32,
                kernel_size=3,
                nb_stacks=1,
                n_layers=4,
                total_epochs=10,
                tv=-1,
                input_type='concat',
                optimizer='Adam',
                dropout=0.,
                units=32,
                n_hidden=1,
                net_type='tcn',
                stateful=False):

    inputs = {
        'lookback': lookback,
        'batch_size': batch_size,
        'nb_filters': nb_filters,
        'kernel_size': kernel_size,
        'nb_stacks': nb_stacks,
        'n_layers': n_layers,
        'dilations': [2**i for i in range(n_layers)],
        'total_epochs': total_epochs,
        'tv': tv,
        'input_type': input_type,
        'net_type': net_type,
        'optimizer': optimizer,
        'stateful': stateful,
        'lookback': lookback,
        'dropout': dropout,
        'units': units,
        'n_hidden': n_hidden,
        'batch_input_shape': (32, lookback, 5)
    }

    return inputs


def create_advanced_net(input_size,
                        net_type='lstm',
                        input_type='concat',
                        **kwargs):

    # net_type is in one of the following
    assert net_type in ['lstm', 'gru', 'tcn']

    # model strategy is one of following
    assert input_type in ['mixed', 'concat']

    # filling in default kwargs values
    if net_type in ['lstm', 'gru']:
        if 'stateful' not in kwargs:
            kwargs['stateful'] = True
        if 'units' not in kwargs:
            kwargs['units'] = 32

    if kwargs['stateful']:
        # should put assertion error here typically
        if 'batch_input_shape' not in kwargs:
            if 'batch_size' not in kwargs:
                kwargs['batch_input_shape'] = (32, input_size[0],
                                               input_size[1])
            else:
                kwargs['batch_input_shape'] = (kwargs['batch_size'],
                                               input_size[0], input_size[1])
    else:
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = None

    if 'n_hidden' not in kwargs:
        kwargs['n_hidden'] = 1

    # make sure no errors are raised for unknown options
    rnn_input_list = ['batch_input_shape', 'stateful', 'units', 'dropout']
    rnn_input = {
        key: value
        for key, value in kwargs.items() if key in rnn_input_list
    }

    tcn_input_list = ['nb_filters', 'kernel_size', 'dilations', 'nb_stacks']
    tcn_input = {
        key: value
        for key, value in kwargs.items() if key in tcn_input_list
    }

    # setting up the input
    time_input = tf.keras.Input(batch_shape=(kwargs['batch_size'],
                                             input_size[0], input_size[1]),
                                name='time_varying')
    fixed_input = tf.keras.Input(batch_shape=(kwargs['batch_size'], 2),
                                 name='time_invariant')

    if kwargs['n_hidden'] == 1:
        return_sequences = False
    else:
        return_sequences = True

    if net_type == 'lstm':

        # adding the first layer
        time_features = tf.keras.layers.LSTM(return_sequences=return_sequences,
                                             **rnn_input)(time_input)

        # adding the remaining layers
        for _ in range(1, kwargs['n_hidden'] - 1):
            time_features = tf.keras.layers.LSTM(
                return_sequences=return_sequences, **rnn_input)(time_features)

        # adding the last layer
        if return_sequences:
            time_features = tf.keras.layers.LSTM(return_sequences=False,
                                                 **rnn_input)(time_features)

    if net_type == 'gru':

        # adding the first layer
        time_features = tf.keras.layers.GRU(return_sequences=return_sequences,
                                            **rnn_input)(time_input)

        # adding the remaining layers
        for _ in range(1, kwargs['n_hidden'] - 1):
            time_features = tf.keras.layers.GRU(
                return_sequences=return_sequences, **rnn_input)(time_features)

        # adding the last layer
        if return_sequences:
            time_features = tf.keras.layers.GRU(return_sequences=False,
                                                **rnn_input)(time_features)

    if net_type == 'tcn':

        # adding the first layer
        time_features = TCN(return_sequences=return_sequences,
                            **tcn_input)(time_input)

        # adding the remaining layers
        for _ in range(1, kwargs['n_hidden'] - 1):
            time_features = TCN(return_sequences=return_sequences,
                                **tcn_input)(time_features)

        # adding the last layer
        if return_sequences:
            time_features = TCN(return_sequences=False,
                                **tcn_input)(time_features)

    if input_type == 'mixed':
        x = time_features
        inputs = [time_input]
    elif input_type == 'concat':
        # concatenating output
        x = tf.keras.layers.concatenate([time_features, fixed_input])
        inputs = [time_input, fixed_input]

    # final dense layer
    power_embedding = tf.keras.layers.Dense(10, activation='relu', name='power_embedding')(x)
    power_pred = tf.keras.layers.Dense(1, name='power_reg')(power_embedding)

    # instantiate the model
    model = tf.keras.Model(
        inputs=inputs,
        outputs=[power_pred],
    )

    model.compile(loss='mean_squared_error',
                  optimizer=kwargs['optimizer'],
                  metrics=[
                      tf.keras.metrics.RootMeanSquaredError(),
                      tf.keras.metrics.MeanAbsolutePercentageError()
                  ])

    return model

def load_model(config, ckpt_path):
    """
    config (dict): contains the configuration of the network we want to build
    ckpt_path (str): If using one of the models from the paper, should contain its name otherwise should contain the path to the model checkpoint
    Returns a model with loaded weights
    """
    inputs = load_inputs(**config)
    model = create_advanced_net((inputs['lookback'], 5), **inputs)
    if not os.path.exists(ckpt_path):
        ckpt_path = './models/{}/cp.ckpt'.format(ckpt_path)
    _ = model.load_weights(ckpt_path).expect_partial()
    
    return (model, inputs)
