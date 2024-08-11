import argparse

from torch.backends import cudnn

from solver import Solver
from utils.utils import *


def str2bool(v):
    return v.lower() in 'true'


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Learning rate of the optimization algorithm
    parser.add_argument('--lr', type=float, default=1e-4)
    # Number of training epoch
    parser.add_argument('--num_epochs', type=int, default=10)
    # The lambda of the loss function
    parser.add_argument('--k', type=int, default=3)
    # Size of the sliding window
    parser.add_argument('--win_size', type=int, default=100)
    # Input dimension (Feature number)
    parser.add_argument('--input_c', type=int, default=38)
    # Output dimension (Feature number)
    parser.add_argument('--output_c', type=int, default=38)
    # Batch size used for data loading -> tuned on VRAM
    parser.add_argument('--batch_size', type=int, default=1024)
    # Type of dataset
    parser.add_argument('--dataset', type=str, choices=['MSL', 'PSM', 'SMAP', 'SMD'], default='MSL')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # Data folder
    parser.add_argument('--data_path', type=str, default='./dataset/MSL')
    # Checkpoints saving config
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    # Anomaly ration percentage for testing
    parser.add_argument('--anomaly_ratio', type=float, default=4.00)
    # size of the ANN inner levels
    parser.add_argument('--d_model', type=int, default=512)
    # Number of Encoder (Anomaly-Attention) layers
    parser.add_argument('--e_layers', type=int, default=3)
    # Number of multi head attention
    parser.add_argument('--n_heads', type=int, default=8)
    # Kernel type
    parser.add_argument('--kernel_type_str', type=str, default='GAUSSIAN')
    # Loss func type
    parser.add_argument('--loss_func_str', type=str, default='MSE_LOSS')
    # Optimizer optimizer
    parser.add_argument('--optimizer_name', type=str, default='ADAM')
    # User LSTM RNN instead of Feed-Forward if not 0
    parser.add_argument('--l_lstm', type=str, default='ADAM')
    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
