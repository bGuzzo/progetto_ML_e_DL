"""
This script performs a grid search to find the optimal hyperparameters for the baseline Transformer Encoder model (with standard self-attention) on the MSL dataset.

This script is analogous to `grid_search.py`, but it is specifically designed for the standard self-attention based Transformer Encoder. The purpose is to find the best hyperparameter configuration for this baseline model, which can then be used for a fair comparison with the Anomaly Transformer.
"""
import itertools
import json
import os.path
import pprint
import time

from self_att_solver import SelfAttSolver

OUT_DIR = 'results'

# This dictionary defines the hyperparameter space for the grid search on the MSL dataset for the standard self-attention model.
MSL_params = {
    'lr':              [1e-4],
    'k':               [3],
    'win_size':        [100],
    'model_save_path': 'checkpoints_self_att',
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
}


def generate_configurations(params):
    """
    Generates all possible hyperparameter configurations from a dictionary of parameter lists.

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
    """

    def __init__(self, dict1):
        self.__dict__.update(dict1)


def train_and_test(dict_config):
    """
    Trains and tests the Transformer Encoder model with a given configuration.

    Args:
        dict_config (dict): A dictionary containing the hyperparameter configuration.

    Returns:
        dict: A dictionary containing the test statistics and the training time.
    """
    config_obj = json.loads(json.dumps(dict_config), object_hook=Obj)
    solver = SelfAttSolver(vars(config_obj))
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
    with open(f"{OUT_DIR}/grid_search_self_att_{time.time()}.log", "w") as log_file:
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
