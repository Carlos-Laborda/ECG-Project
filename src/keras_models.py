import keras
from keras import Input, layers, models, optimizers, losses, metrics, regularizers

# ------------------------------------------------------
# Models for ECG Classification
# ------------------------------------------------------
### CNNs
def cnn_overfit_simple(input_length=10000):
    """1D CNN model designed for overfitting tests (2 participants)"""
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(input_length, 1)),
        
        # Convolutional layer
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.BatchNormalization(),
        
        # Flaten and Dense layers
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    
    return model

def cnn_overfit(input_length=10000):
    """ it overfits the data 20 participants"""
    model = models.Sequential([
        layers.Input(shape=(input_length, 1)),
        layers.BatchNormalization(),
        
        # First conv block
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.BatchNormalization(),
        
        # Second conv block
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.BatchNormalization(),
        
        # Third conv block for increased capacity
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.BatchNormalization(),
        
        # Global pooling summarizes features without overcompressing spatial info
        #layers.GlobalAveragePooling1D(),
        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    
    return model

def baseline_1DCNN(input_length=10000):
    """sets the baseline performance for the 1D CNN model"""
    inputs = layers.Input(shape=(input_length, 1))
    x = layers.BatchNormalization()(inputs)
    
    # Convolution Block 1
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    # Convolution Block 2
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    # Convolution Block 3
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers with dropout
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

def improved_1DCNN(input_length=10000):
    inputs = layers.Input(shape=(input_length, 1))
    x = layers.BatchNormalization()(inputs)
    
    # Convolution Block 1 - smaller filters for local patterns
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.1)(x)  # Feature map dropout
    
    # Convolution Block 2 - medium filters
    x = layers.Conv1D(64, 11, activation='relu', padding='same')(x)
    x = layers.Conv1D(64, 11, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.1)(x)
    
    # Convolution Block 3 - larger filters for context
    x = layers.Conv1D(128, 17, activation='relu', padding='same')(x)
    x = layers.Conv1D(128, 17, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    
    # Use a schedule to reduce learning rate
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=337,
        decay_rate=0.9)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

### LSTMs
def baseline_LSTM(input_shape=(10000, 1)):
    """
    Build a simple LSTM model for binary ECG classification.
    
    Args:
        input_shape (tuple): Shape of input signal (window_length, channels)
        
    Returns:
        keras.Model: Compiled Keras model with LSTM architecture
    """
    model = models.Sequential([
        # LSTM layers
        layers.LSTM(64, return_sequences=True, 
                   input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Second LSTM layer
        layers.LSTM(32),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Dense layers for classification
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()]
    )
    
    return model

### Neural Networks
def neural_network(input_shape=(10000, 1)):
    """Build and compile the neural network to predict the species of a penguin."""

    model = models.Sequential([
        # Reshape input to be 1D
        layers.Reshape((input_shape,), input_shape=(input_shape,)),
        
        # Dense layers
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        
        # Output layer for binary classification
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01), 
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    return model