"""
This script is the main entry point for training and testing the Anomaly Transformer model.

It uses `argparse` to parse command-line arguments, which allows for flexible configuration of the model, training process, and dataset. The script then initializes a `Solver` instance and calls the appropriate method (`train` or `test`) based on the specified mode.
"""
import argparse

from torch.backends import cudnn

from solver import Solver
from utils.utils import *


def str2bool(v):
    """
    Converts a string to a boolean value.

    Args:
        v (str): The input string.

    Returns:
        bool: True if the string is "true" (case-insensitive), False otherwise.
    """
    return v.lower() in 'true'


def mkdir(directory):
    """
    Creates a directory if it does not already exist.

    Args:
        directory (str): The path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(config):
    """
    The main function for training and testing the Anomaly Transformer model.

    This function initializes the `Solver` and starts the training or testing process based on the provided configuration.

    Args:
        config (argparse.Namespace): A namespace object containing the configuration parameters.

    Returns:
        Solver: The solver instance.
    """
    # Set the cuDNN benchmark flag for performance optimization.
    cudnn.benchmark = True
    # Create the model save directory if it doesn't exist.
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    # Start training or testing based on the specified mode.
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # The learning rate for the optimizer.
    parser.add_argument('--lr', type=float, default=1e-4)
    # The number of training epochs.
    parser.add_argument('--num_epochs', type=int, default=10)
    # The lambda parameter for the loss function, balancing the reconstruction and association discrepancy losses.
    parser.add_argument('--k', type=int, default=3)
    # The size of the sliding window for creating time series segments.
    parser.add_argument('--win_size', type=int, default=100)
    # The number of input features.
    parser.add_argument('--input_c', type=int, default=38)
    # The number of output features.
    parser.add_argument('--output_c', type=int, default=38)
    # The batch size for training and testing.
    parser.add_argument('--batch_size', type=int, default=1024)
    # The name of the dataset to use.
    parser.add_argument('--dataset', type=str, choices=['MSL', 'PSM', 'SMAP', 'SMD'], default='MSL')
    # The mode of operation: 'train' or 'test'.
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # The path to the dataset files.
    parser.add_argument('--data_path', type=str, default='./dataset/MSL')
    # The path to save the model checkpoints.
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    # The anomaly ratio for the dataset, used for evaluation.
    parser.add_argument('--anomaly_ratio', type=float, default=4.00)
    # The dimensionality of the model's hidden states.
    parser.add_argument('--d_model', type=int, default=512)
    # The number of encoder layers in the Anomaly Transformer.
    parser.add_argument('--e_layers', type=int, default=3)
    # The number of heads in the multi-head attention mechanism.
    parser.add_argument('--n_heads', type=int, default=8)
    # The type of kernel to use for the prior-association.
    parser.add_argument('--kernel_type_str', type=str, default='GAUSSIAN')
    # The type of loss function to use.
    parser.add_argument('--loss_func_str', type=str, default='MSE_LOSS')
    # The name of the optimizer to use.
    parser.add_argument('--optimizer_name', type=str, default='ADAM')
    # The number of layers in the optional LSTM network. If 0, a feed-forward network is used.
    parser.add_argument('--l_lstm', type=str, default='ADAM')
    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)