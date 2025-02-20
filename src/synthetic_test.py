import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import keras
from keras import layers, models, optimizers
import matplotlib.pyplot as plt

def cnn_overfit(input_length=10000):
    """1D CNN model"""
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(input_length, 1)),
        
        # Convolutional layer
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        
        # Global Average Pooling 
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    
    return model

def generate_sine_ecg(num_samples=10, length=10000, freq=1.0):
    """Generate synthetic ECG-like data"""
    t = np.linspace(0, 2*np.pi*freq, length)
    data = []
    labels = []
    
    for i in range(num_samples):
        # Baseline: simple sine wave
        data.append(np.sin(t + i*0.1))
        labels.append(0)
        
        # Stress: higher amplitude sine wave
        data.append(2.0 * np.sin(t + i*0.1))
        labels.append(1)
    
    data = np.array(data)[..., None]  # shape: (2*num_samples, length, 1)
    labels = np.array(labels)
    
    return data, labels

def plot_samples(X, y, num_samples=2):
    """Plot sample windows from each class"""
    plt.figure(figsize=(15, 5))
    
    for i in range(num_samples):
        baseline_idx = np.where(y == 0)[0][i]
        stress_idx = np.where(y == 1)[0][i]
        
        plt.subplot(2, num_samples, i + 1)
        plt.plot(X[baseline_idx, :, 0])
        plt.title(f'Baseline Sample {i+1}')
        
        plt.subplot(2, num_samples, i + num_samples + 1)
        plt.plot(X[stress_idx, :, 0])
        plt.title(f'Stress Sample {i+1}')
    
    plt.tight_layout()
    plt.show()

def run_synthetic_test():
    """Run overfitting test with synthetic data"""
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = generate_sine_ecg(num_samples=5)  # 10 total samples (5 per class)
    
    print("\n=== Data Shapes ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Plot samples
    plot_samples(X, y)
    
    # Create model
    print("\nInitializing model...")
    model = cnn_overfit(input_length=X.shape[1])
    
    # Custom callback to print progress
    class PrintProgress(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/100], Loss: {logs["loss"]:.3f}, Accuracy: {logs["binary_accuracy"]:.3f}')
    
    # Training
    print("\nTraining model on synthetic data...")
    history = model.fit(
        X, y,
        epochs=100,
        batch_size=2,
        verbose=0,
        callbacks=[PrintProgress()],
        validation_data=(X, y)  # Using same data for validation to check overfitting
    )
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['binary_accuracy'], label='Accuracy')
    plt.title('Model Accuracy on Synthetic Data')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Final evaluation
    y_pred = (model.predict(X) > 0.5).astype(int)
    
    print("\n=== Final Results ===")
    print("Predictions:", y_pred.flatten())
    print("True labels:", y)
    
    # Check if perfectly memorized
    is_memorized = np.array_equal(y_pred.flatten(), y)
    print(f"\nPerfectly memorized: {is_memorized}")

if __name__ == "__main__":
    run_synthetic_test()