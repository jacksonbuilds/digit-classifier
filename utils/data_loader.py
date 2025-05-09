# utils/data_loader.py
import tensorflow as tf

def load_mnist(normalize: bool = True):
    """Loads and optionally normalizes the MNIST dataset."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    return (X_train, y_train), (X_test, y_test)
