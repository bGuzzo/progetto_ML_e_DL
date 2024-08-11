import itertools
import json
import os.path
import pprint
import time

from self_att_solver import SelfAttSolver

OUT_DIR = 'results'

# Params for Classical Self-Attention model
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
    list_keys = [k for k, v in params.items() if isinstance(v, list)]
    list_values = [params[k] for k in list_keys]
    all_combinations = itertools.product(*list_values)

    for combo in all_combinations:
        config = params.copy()
        for i, key in enumerate(list_keys):
            config[key] = combo[i]
        yield config


class Obj:

    def __init__(self, dict1):
        self.__dict__.update(dict1)


def train_and_test(dict_config):
    config_obj = json.loads(json.dumps(dict_config), object_hook=Obj)
    solver = SelfAttSolver(vars(config_obj))
    start_time = time.time()
    solver.train()
    end_time = time.time()
    test_stats = solver.test()
    test_stats['train_time'] = end_time - start_time
    return test_stats


if __name__ == '__main__':
    # Create a list to hold the configurations
    configurations = list(generate_configurations(MSL_params))
    results = []
    print(f'Start Grid search over {len(configurations)} possible hyperparam combination')

    # Print each configuration
    for config in configurations:
        print(f"Using config {config}")
        metric = train_and_test(config)
        results.append({
            'config': config,
            'metric': metric
        })
        print(f'Metrics {metric} of config {config}')

    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    with open(f"{OUT_DIR}/grid_search_self_att_{time.time()}.log", "w") as log_file:
        print(f'Grid search results over {len(configurations)} possible hyperparams:')
        print(f'Grid search results over {len(configurations)} possible hyperparams:', file=log_file)
        pprint.pprint(results)
        pprint.pprint(results, stream=log_file)
        print('Best configuration and metrics:')
        print('\n', file=log_file)
        print('Best configuration and metrics:', file=log_file)
        best_config = sorted(results, key=lambda r: r['metric']['f_score'], reverse=True)[0]
        pprint.pprint(best_config)
        pprint.pprint(best_config, stream=log_file)