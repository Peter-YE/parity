import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def envelope_extraction(signal, n_node, n_step):
    """Extract the envelope of the signal using the Hilbert transform."""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    envelope = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope))
    indices = np.linspace(0, len(envelope) - 1, n_node * n_step, dtype=int)
    envelope = envelope[indices]
    return envelope


def parity_benchmark(step_val, n):
    """Calculate the parity of the input signal at each step."""
    parity = np.ones(step_val.size)
    for p in range(parity.size):
        if p >= n - 1:  # Ensure we have enough history to calculate parity
            parity[p] = np.prod(step_val[p - n + 1:p + 1])
        else:
            parity[p] = 0  # Parity is undefined for the first n-1 steps
    return parity

def load_data():
    # Load data
    step_time = np.loadtxt("step_time.csv", delimiter=",")
    step_val = np.loadtxt("step_val.csv", delimiter=",")

    time = np.loadtxt("time.csv", delimiter=",")
    z1 = np.loadtxt("z1.csv", delimiter=",")

    return step_time, step_val, time, z1

def create_dataset(n_node, n_step, envelope, parity):
    x_data = np.zeros((n_step, n_node))
    y_data = np.zeros(n_step)

    for i in range(n_step):
        x_data[i] = envelope[i * n_node:i * n_node + n_node]  # Use a window of n_node values
        y_data[i] = parity[i]  # Use parity at the current step
    y_data = (y_data + 1) / 2
    # Convert y_data to one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y_data, num_classes=2)

    return x_data, y_onehot

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
        tf.keras.layers.Dense(64, input_shape=(n_node,), activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=2000, verbose=1)
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_accuracy}")

    return model
def main():
    print("Running...")

    step_time, step_val, time, z1 = load_data()



    # Calculate parity at each step
    n = 2  # Parity order
    parity = parity_benchmark(step_val, n)
    n_node = 50  # Number of past steps to use as input features
    n_step = len(parity)  # Number of samples after creating windows

    # Extract envelope
    envelope = envelope_extraction(z1, n_node, n_step)
    # Create dataset with n_node


    x_data, y_data = create_dataset(n_node, n_step, envelope, parity)
    # Create TensorFlow dataset

    # Define split ratio
    train_ratio = 0.8  # 80% training, 20% testing
    batch_size = 32

    train_dataset, test_dataset = split_dataset(x_data, y_data, train_ratio, batch_size)

    model = train_model(train_dataset, test_dataset, n_node)

    # Predict and plot
    predictions = model.predict(x_data)
    predictions = np.argmax(predictions, axis=1)  # Convert one-hot to class labels
    # convert y_data from one-hot to class labels
    y_data = np.argmax(y_data, axis=1)

    plot_data(time, y_data, predictions)


if __name__ == '__main__':
    main()