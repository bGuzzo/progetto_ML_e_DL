"""
This module provides various utility functions for the project.

It includes functions for handling PyTorch tensors, creating directories, and generating plots for visualizing the results.
"""
import os

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


def to_var(x, volatile=False):
    """
    Wraps a tensor in a `Variable` and moves it to the GPU if available.

    Note: The `Variable` class is deprecated in recent PyTorch versions, and the `volatile` parameter has no effect. This function is kept for compatibility with older code.

    Args:
        x (torch.Tensor): The input tensor.
        volatile (bool, optional): This parameter is deprecated and has no effect. Defaults to False.

    Returns:
        torch.Tensor: The tensor on the GPU if available, otherwise the original tensor.
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    """
    Creates a directory if it does not already exist.

    Args:
        directory (str): The path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_sampled_plot(test_energy, thresh, sampling_rate=10, title="Heist Plot (sampled)"):
    """
    Generates and displays a plot of the anomaly scores (test energy) with a threshold line.

    This function is useful for visualizing the anomaly detection results. It uses sampling to handle large datasets efficiently.

    Args:
        test_energy (numpy.ndarray): An array of anomaly scores.
        thresh (float): The anomaly threshold.
        sampling_rate (int, optional): The rate at which to sample the data points for plotting. Defaults to 10.
        title (str, optional): The title of the plot. Defaults to "Heist Plot (sampled)".
    """
    print(f'Plotting an array of {len(test_energy)} data points')

    # Apply sampling to the data.
    sampled_data = test_energy[::sampling_rate]

    # Create the plot.
    fig, ax = plt.subplots()

    # Plot the anomaly scores as a line graph.
    ax.plot(sampled_data, color='red', linewidth=1)

    # Plot the threshold line.
    ax.axhline(y=thresh, color='blue', linestyle='--', label=f'Threshold: {thresh:.7f}')

    # Add gridlines for better readability.
    ax.grid(True, linestyle='--')

    # Set the title and labels.
    ax.set_title(title)
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Test Energy')

    # Display the legend.
    ax.legend()

    # Show the plot.
    plt.tight_layout()
    plt.show()