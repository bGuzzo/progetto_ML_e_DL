import os

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_sampled_plot(test_energy, thresh, sampling_rate=10, title="Heist Plot (sampled)"):
    print(f'Plotting an array of {len(test_energy)} data points')

    # Apply sampling
    sampled_data = test_energy[::sampling_rate]

    # Create the plot
    fig, ax = plt.subplots()

    # Plot the data as a line graph
    ax.plot(sampled_data, color='red', linewidth=1)

    # Plot the threshold line
    ax.axhline(y=thresh, color='blue', linestyle='--', label=f'Threshold: {thresh:.7f}')

    # Add gridlines
    ax.grid(True, linestyle='--')

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Test Energy')

    # Display the legend (this is the key addition)
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
