import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import hilbert



def test():
    test1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    test2 = test1[::2]
    print(test2)
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




    n = 5
    parity = parity_benchmark(step_val, n)
    n_data = time.size // parity.size
    n_step = step_time.size
    # expand parity to the size of time
    print("time size", time.size)
    print("n_data", n_data)
    print("n_step", n_step)
    print("parity size", parity.size)
    parity_plot = np.repeat(parity, n_data)
    #parity_plot *= 1e-25
    print(parity_plot.size)

    # plot
    plt.figure()
    plt.plot(time, envelope)
    plt.plot(time, parity_plot)
    plt.show()




    # create data set
    n_node = n_data // 100
    x_data = np.zeros((n_step, n_node))
    y_data = np.zeros(n_step)

    points = envelope[::100]
    for i in range(n_step):
        x_data[i] = points[i*n_node:(i+1)*n_node]
        y_data[i] = parity[i]
    y_data = (y_data+1)/2
    #y_data = np.expand_dims(y_data, axis=1)

    y_onehot = tf.keras.utils.to_categorical(y_data, num_classes=2)
    # x_data = x_data[10:190,:]
    # y_onehot = y_onehot[10:190,:]
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_onehot))
    batch_size = 10
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("xshape",x_data.shape)
    print("yshape",y_onehot.shape)
    #print(y_onehot)



    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2, input_shape=(n_node,), activation='softmax'),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    model.fit(dataset, epochs=1000)

if __name__ == '__main__':
    main()

