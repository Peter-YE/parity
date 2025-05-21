import numpy as np
import utility
import reservoir
import matplotlib.pyplot as plt

def main():
    print("Running...")

    step_time, step_val, time, z1 = reservoir.reservoir()
    plt.figure()
    for n in range(2,5):
        # Calculate parity at each step
        n_node = 50  # Number of past steps to use as input features
        # Define split ratio
        train_ratio = 0.8  # 80% training, 20% testing
        batch_size = 32

        parity = utility.parity_benchmark(step_val, n)
        n_step = len(parity)  # Number of samples after creating windows


        # Extract envelope
        envelope = utility.envelope_extraction(z1, n_node, n_step)
        # Create dataset with n_node

        x_data, y_data = utility.create_dataset(n_node, n_step, envelope, parity)
        # Create TensorFlow dataset

        train_dataset, test_dataset = utility.split_dataset(x_data, y_data, train_ratio, batch_size)

        model = utility.train_model(train_dataset, test_dataset, n_node)

        # Predict and plot
        predictions = model.predict(x_data)
        # predictions = np.argmax(predictions, axis=1)  # Convert one-hot to class labels
        predictions = predictions[:,1]
        # convert y_data from one-hot to class labels
        y_data = np.argmax(y_data, axis=1)

        plt.subplot(3, 1, n - 1)

        plot_time = time[500:1000]
        parity = np.repeat(parity, 10)
        predictions = predictions * 2 - 1
        predictions = np.repeat(predictions, 10)

        # Plot target signal
        plt.plot(plot_time, parity[500:1000], label='target', color='orange', linewidth=2)

        # Plot classification result
        plt.plot(plot_time, predictions[500:1000], label='classification', linestyle='--', color='C0',
                 linewidth=2)

        # Remove top and right spines for a cleaner look
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Make ticks a bit smaller and more elegant
        plt.tick_params(axis='both', labelsize=14, direction='out', length=6, width=1)

        # Add grid for better readability
        plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)

        # Remove x-axis labels for upper subplots
        if n - 1 != 3:
            plt.tick_params(labelbottom=False)

    plt.show()
    # utility.plot_data(time, y_data, predictions)


if __name__ == '__main__':
    main()
