"""
This script performs a grid search to find the optimal hyperparameters for the Anomaly Transformer model on the MSL dataset.

It defines a set of hyperparameters to search over, generates all possible combinations of these hyperparameters, and then trains and evaluates the model for each combination. The results of the grid search are saved to a log file, and the best configuration based on the F-score is printed to the console.
"""
import itertools
import json
import os
import pprint
import time

from model.kernel import KernelType
from model.loss_func import LossFunc
from model.optimizer import OptimizerAlg
from solver import Solver

OUT_DIR = 'results'

# This dictionary defines the hyperparameter space for the grid search on the MSL dataset.
# The keys are the hyperparameter names, and the values are lists of possible values to try.
MSL_params = {
    'lr':              [1e-4],
    'k':               [3],
    'win_size':        [100],
    'model_save_path': 'checkpoints',
    'dataset':         'MSL',
    'data_path':       'dataset/MSL',
    'anomaly_ratio':   [1],
    'num_epochs':      [2],
    'batch_size':      [128],
    'input_c':         55,
    'output_c':        55,
    'd_model':         [64],
    'e_layers':        [3],
    'n_heads':         [8],
    'kernel_type_str': [KernelType.GAUSSIAN.name],
    'loss_func_str':   [LossFunc.MSE_LOSS.name],
    'optimizer_name':  [OptimizerAlg.ADAM.name],
    'l_lstm':          [0]
}


def generate_configurations(params):
    """
    Generates all possible hyperparameter configurations from a dictionary of parameter lists.

    This function takes a dictionary where the values are lists of possible parameter values, and it yields all possible combinations of these parameters.

    Args:
        params (dict): A dictionary where keys are parameter names and values are lists of possible values.

    Yields:
        dict: A dictionary representing a single hyperparameter configuration.
    """
    list_keys = [k for k, v in params.items() if isinstance(v, list)]
    list_values = [params[k] for k in list_keys]
    all_combinations = itertools.product(*list_values)

    for combo in all_combinations:
        config = params.copy()
        for i, key in enumerate(list_keys):
            config[key] = combo[i]
        yield config


class Obj:
    """
    A helper class to convert a dictionary to an object.

    This allows accessing the dictionary keys as attributes of the object.
    """

    def __init__(self, dict1):
        self.__dict__.update(dict1)


def train_and_test(dict_config):
    """
    Trains and tests the Anomaly Transformer model with a given configuration.

    Args:
        dict_config (dict): A dictionary containing the hyperparameter configuration.

    Returns:
        dict: A dictionary containing the test statistics (e.g., accuracy, precision, recall, F-score) and the training time.
    """
    config_obj = json.loads(json.dumps(dict_config), object_hook=Obj)
    solver = Solver(vars(config_obj))
    start_time = time.time()
    solver.train()
    end_time = time.time()
    test_stats = solver.test()
    test_stats['train_time'] = end_time - start_time
    return test_stats


if __name__ == '__main__':
    # Generate all possible hyperparameter configurations.
    configurations = list(generate_configurations(MSL_params))
    results = []
    print(f'Start Grid search over {len(configurations)} possible hyperparam combination')

    # Iterate through each configuration, train and test the model, and store the results.
    for config in configurations:
        print(f"Using config {config}")
        metric = train_and_test(config)
        results.append({
            'config': config,
            'metric': metric
        })
        print(f'Metrics {metric} of config {config}')

    # Create the output directory if it doesn't exist.
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    # Save the grid search results to a log file.
    with open(f"{OUT_DIR}/grid_search_{time.time()}.log", "w") as log_file:
        print(f'Grid search results over {len(configurations)} possible hyperparams:')
        print(f'Grid search results over {len(configurations)} possible hyperparams:', file=log_file)
        pprint.pprint(results)
        pprint.pprint(results, stream=log_file)
        print('Best configuration and metrics:')
        print('\n', file=log_file)
        print('Best configuration and metrics:', file=log_file)
        # Find and print the best configuration based on the F-score.
        best_config = sorted(results, key=lambda r: r['metric']['f_score'], reverse=True)[0]
        pprint.pprint(best_config)
        pprint.pprint(best_config, stream=log_file)