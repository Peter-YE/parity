import numpy as np
from matplotlib.pyplot import figure

import utility
import reservoir
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    print("Running...")

    step_time, step_val, time, v1 = reservoir.reservoir()
    plt.figure()
    for order in range(2, 5):
        # Calculate parity at each step
        n_node = 100  # Number of samples to use as input features
        # Define split ratio
        train_ratio = 0.8  # 80% training, 20% testing
        batch_size = 32

        parity = utility.parity_benchmark(step_val, order)
        n_step = len(parity)  # Number of samples after creating windows

        # Extract envelope
        envelope = utility.envelope_extraction(v1, n_node, n_step)
        # Create dataset with n_node

        x_data, y_data = utility.create_dataset(n_node, n_step, envelope, parity)
        x_train, y_train, x_test, y_test = utility.split_dataset_ridge(x_data, y_data, train_ratio)
        w = utility.ridge_regression(x_train, y_train, x_test, y_test)
        # Create TensorFlow dataset
        prediction_reg = x_data @ w

        train_dataset, test_dataset = utility.split_dataset(x_data, y_data, train_ratio, batch_size)

        model = utility.train_model(train_dataset, test_dataset, n_node)

        # Predict and plot

        # # Remove comment to plot benchmarking results
        predictions = model.predict(x_data)
        predictions = prediction_reg[:, 1]
        plt.subplot(3, 1, order - 1)
        parity = np.repeat(parity, 10)
        predictions = predictions * 2 - 1
        predictions = np.repeat(predictions, 10)
        # cap the predictions to -1 and 1
        predictions = np.clip(predictions, -1, 1)

        # Plot target signal
        plt.plot(time[500:1000], parity[500:1000], label='target', color='orange', linewidth=2)
        plt.plot(time[500:1000], predictions[500:1000], label='classification', linestyle='--', color='C0',linewidth=2)
        # Remove top and right spines for a cleaner look
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylim(top=1.2)
        plt.ylim(bottom=-1.2)

        # Make ticks a bit smaller and more elegant
        plt.tick_params(axis='both', labelsize=14, direction='out', length=6, width=1)

        # Add grid for better readability
        plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)

        # Remove x-axis labels for upper subplots
        if order - 1 != 3:
            plt.tick_params(labelbottom=False)

        # # Remove comment to plot comparison between signal and envelope
        # plt.figure(figsize=(14, 8))
        # indices = np.linspace(0, len(v1) - 1, n_node * n_step, dtype=int)
        # v1_plot = v1[indices]
        # plt.plot(time[500:1000], v1_plot[500:1000], label='signal', color='C0', linewidth=2)
        #
        #
        # # Plot classification result
        # plt.plot(time[500:1000], envelope[500:1000]/16000, label='envelope', linestyle='--', color='orange',
        #          linewidth=5)
        # plt.xlabel('Time (s)', fontsize=30)
        # plt.ylabel('Displacement Velocity (m/s)', fontsize=30)
        # plt.legend(loc='best', fontsize=30)
        # plt.tick_params(axis='both', labelsize=30)
        # ax1 = plt.gca()  # Get current axes
        # ax1.xaxis.offsetText.set_fontsize(30)
        # ax1.yaxis.offsetText.set_fontsize(30)




    plt.show()


if __name__ == '__main__':
    main()
