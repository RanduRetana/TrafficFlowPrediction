"""
Defination of NN model
"""
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, Input
from keras.models import Sequential, Model


def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model



def get_saes(layers):
    """
    Build SAEs model.
    
    Arguments:
    layers: A list of integers, the number of units in each layer
    """
    models = []
    for i in range(1, len(layers) - 1):
        inputs = Input(shape=(layers[i-1],))
        encoded = Dense(layers[i], activation='sigmoid', name=f'hidden{i}')(inputs)
        decoded = Dense(layers[i-1], activation='sigmoid')(encoded)
        
        autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        models.append((autoencoder, encoder))
    
    # The final model
    saes = Sequential()
    saes.add(Input(shape=(layers[0],)))
    for i in range(1, len(layers) - 1):
        saes.add(Dense(layers[i], activation='sigmoid', name=f'hidden{i}'))
    saes.add(Dense(layers[-1], activation='linear'))  # Output layer
    
    return models, saes
