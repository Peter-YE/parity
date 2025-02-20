import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def envelope_extraction(signal):
    """Extract the envelope of the signal using the Hilbert transform."""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
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


def main():
    print("Running...")

    # Load data
    step_time = np.loadtxt("step_time.csv", delimiter=",")
    step_time = np.delete(step_time, 0)
    step_val = np.loadtxt("step_val.csv", delimiter=",")
    step_val = np.delete(step_val, 0)

    time = np.loadtxt("time.csv", delimiter=",")
    time = np.delete(time, 0)
    z1 = np.loadtxt("z1.csv", delimiter=",")
    z1 = np.delete(z1, 0)

    # Extract envelope
    envelope = envelope_extraction(z1)
    envelope = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope))

    # Calculate parity at each step
    n = 2  # Parity order
    parity = parity_benchmark(step_val, n)

    # Create dataset with n_node
    n_node = 50  # Number of past steps to use as input features
    n_step = len(parity) - n_node + 1  # Number of samples after creating windows

    x_data = np.zeros((n_step, n_node))
    y_data = np.zeros(n_step)

    for i in range(n_step):
        x_data[i] = envelope[i:i + n_node]  # Use a window of n_node values
        y_data[i] = parity[i + n_node - 1]  # Use parity at the current step
    y_data = (y_data + 1) / 2
    # Convert y_data to one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y_data, num_classes=2)

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_onehot))
    batch_size = 32
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create and train model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=(n_node,), activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(dataset, epochs=10000, verbose=1)

    # Predict and plot
    predictions = model.predict(x_data)
    predictions = np.argmax(predictions, axis=1)  # Convert one-hot to class labels
    n_data = time.size // parity.size

    plt.figure()
    plot_time = time[1:500]
    #plt.plot(time[n_node - 1:], envelope[n_node - 1:], label='Envelope')
    parity = np.repeat(parity,10)
    predictions = predictions*2 - 1
    predictions = np.repeat(predictions,10)
    plt.plot(plot_time,parity[1:500], label='Parity')
    plt.plot(plot_time,predictions[1:500], label='Predictions', linestyle='--')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()