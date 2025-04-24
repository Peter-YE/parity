import numpy as np
import utility
import reservoir

def main():
    print("Running...")

    step_time, step_val, time, z1 = reservoir.reservoir()

    # Calculate parity at each step
    n = 2  # Parity order
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
    predictions = np.argmax(predictions, axis=1)  # Convert one-hot to class labels
    # convert y_data from one-hot to class labels
    y_data = np.argmax(y_data, axis=1)

    utility.plot_data(time, y_data, predictions)


if __name__ == '__main__':
    main()
