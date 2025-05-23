import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import numpy as np


def envelope_extraction(signal, n_node, n_step):
    """Extract the envelope of the signal using the Hilbert transform."""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    envelope = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope))
    indices = np.linspace(0, len(envelope) - 1, n_node * n_step, dtype=int)
    envelope = envelope[indices]
    return envelope


def parity_benchmark(step_val: np.ndarray, n: int) -> np.ndarray:
    """Calculate parity of the input signal with rolling window."""
    parity = np.zeros(step_val.size)
    rolling_window = np.lib.stride_tricks.sliding_window_view(step_val, n)
    parity[n - 1:] = rolling_window.prod(axis=1)

    return parity


def load_data():
    # Load data
    step_time = np.loadtxt("step_time.csv", delimiter=",")
    step_val = np.loadtxt("step_val.csv", delimiter=",")

    time = np.loadtxt("time.csv", delimiter=",")
    z1 = np.loadtxt("z1.csv", delimiter=",")

    return step_time, step_val, time, z1


def create_dataset(n_node: int, n_step: int, envelope: np.ndarray, parity: np.ndarray):
    x_data = envelope.reshape(n_step, n_node)
    y_data = tf.keras.utils.to_categorical((parity + 1) // 2, num_classes=2)
    return x_data, y_data


def split_dataset_ridge(x_data, y_data, train_ratio):
    split_index = int(len(x_data) * train_ratio)  # Compute split index

    # Shuffle data (important for training)
    indices = np.random.permutation(len(x_data))
    x_data, y_data = x_data[indices], y_data[indices]  # Shuffle both inputs and labels

    # Split data
    x_train, x_test = x_data[:split_index], x_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]

    return x_train, y_train, x_test, y_test


def split_dataset(x_data, y_data, train_ratio, batch_size):
    split_index = int(len(x_data) * train_ratio)  # Compute split index

    # Shuffle data (important for training)
    indices = np.random.permutation(len(x_data))
    x_data, y_data = x_data[indices], y_data[indices]  # Shuffle both inputs and labels

    # Split data
    x_train, x_test = x_data[:split_index], x_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


def plot_data(time, parity, predictions):
    plt.figure()
    plot_time = time[500:1000]
    parity = np.repeat(parity, 10)
    # predictions = predictions * 2 - 1
    predictions = np.repeat(predictions, 10)
    plt.plot(plot_time, parity[500:1000], label='Parity')
    plt.plot(plot_time, predictions[500:1000], label='Predictions', linestyle='--')
    plt.legend()
    plt.show()


def train_model(train_dataset, test_dataset, n_node):
    # Create and train model
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Dense(2, input_shape=(n_node,), activation='softmax'),
        tf.keras.Input(shape=(n_node,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=2000, verbose=0)
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"NN accuracy: {test_accuracy}")

    return model


def ridge_regression(x_train, y_train, x_test, y_test):
    w = np.linalg.pinv(x_train) @ y_train
    y_pred = x_test @ w
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_pred == y_test)
    print(f"Regression accuracy: {accuracy}")
