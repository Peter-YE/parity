import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import hilbert






def parity_benchmark(step_val, n):
    step_val = step_val * 2 - 1
    # parity benchmark, e.g. when n = 3, the parity is $\prod_{i=0}^2 u(t-(i+\tau) T)$, u is the step function from input
    parity = np.ones(step_val.size)
    for p in range(parity.size - n + 1):
        for i in range(n):
            parity[p] *= step_val[p + i]
    return parity


def main():
    print("running")
    step_time = np.loadtxt("step_time.csv", delimiter=",")
    step_time = np.delete(step_time, 0)
    step_val = np.loadtxt("step_val.csv", delimiter=",")
    step_val = np.delete(step_val, 0)

    time = np.loadtxt("time.csv", delimiter=",")
    time = np.delete(time, 0)
    z1 = np.loadtxt("z1.csv", delimiter=",")
    z1 = np.delete(z1, 0)
    analytic_signal = hilbert(z1)
    envelope = np.abs(analytic_signal)
    envelope = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope))
    print(envelope.shape)



# parity order
    n = 2
    parity = parity_benchmark(step_val, n)
    n_data = time.size // parity.size
    n_step = step_time.size
    # expand parity to the size of time
    print("time size", time.size)
    print("n_data", n_data)
    print("n_step", n_step)
    print("parity size", parity.size)
    parity_plot = np.repeat(parity, n_data)
    parity_plot = parity_plot*0.5 - 1
    print(parity_plot.size)







    # create data set
    n_gap = 50
    n_node = n_data // n_gap
    x_data = np.zeros((n_step, n_node))
    y_data = np.zeros(n_step)

    points = envelope[::n_gap]
    for i in range(n_step):
        x_data[i] = points[i*n_node:(i+1)*n_node]
        y_data[i] = parity[i]
    y_data = (y_data+1)/2
    #y_data = np.expand_dims(y_data, axis=1)

    y_onehot = tf.keras.utils.to_categorical(y_data, num_classes=2)
    # x_data = x_data[10:190,:]
    # y_onehot = y_onehot[10:190,:]
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_onehot))
    batch_size = 5000
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("xshape",x_data.shape)
    print("yshape",y_onehot.shape)
    #print(y_onehot)


    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2, input_shape=(n_node,), activation='relu'),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(dataset, epochs=10000, verbose=1)
    plt.plot(history.history['loss'], label='Training Loss')
    predictions = model.predict(x_data)

    predictions = predictions[:,0]
    predictions = predictions * 0.5 - 1
    predictions = np.repeat(predictions, n_data)

    # plot
    plt.figure()
    scale = 1000
    plt.plot(time[:len(time)//scale], envelope[:len(envelope)//scale])
    plt.plot(time[:len(time)//scale], parity_plot[:len(parity_plot)//scale])
    plt.plot(time[:len(time)//scale],predictions[:len(predictions)//scale])

    # plt.plot(time, envelope)
    # plt.plot(time, parity_plot)
    # plt.plot(time, predictions)
    plt.show()

if __name__ == '__main__':
    main()

