import os
import time
import argparse
import collections
import multiprocessing

from pathlib import Path

import numpy as np

from nat_rl.utils.evaluation import evaluate_grid, SUCCESS_RATES_FNAME
from nat_rl.utils.env_utils import make_GC_pick_fruit_env


env_fns = {
    'gc_pick_fruit': make_GC_pick_fruit_env
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gc_pick_fruit')
    args = parser.parse_args()
    return args

def main(args, mp_ctx):
    assert args.env in env_fns

    env_fn_kwargs = {
        'test_dataset': True,
        'env_kwargs': {
            'goal_format': 'clip_lang', # NOTE: change this if needed
            'load_goals': True
        }
    }

    # runs = [
    #     'logs/bc/saved_models/gc_pick_fruit_CNN_seed=7_ts-1651650105',
    #     'logs/bc/saved_models/gc_pick_fruit_CNN_seed=55_ts-1651642267',
    #     'logs/bc/saved_models/gc_pick_fruit_CNN_seed=59_ts-1651673775',
    #     'logs/bc/saved_models/gc_pick_fruit_CNN_seed=62_ts-1651665986',
    #     'logs/bc/saved_models/gc_pick_fruit_CNN_seed=88_ts-1651658028'
    # ]

    # runs = [
    #     'logs/bc/saved_models/gc_pick_fruit_CLIP_seed=7_ts-1651651303',
    #     'logs/bc/saved_models/gc_pick_fruit_CLIP_seed=55_ts-1651646657',
    #     'logs/bc/saved_models/gc_pick_fruit_CLIP_seed=59_ts-1651665438',
    #     'logs/bc/saved_models/gc_pick_fruit_CLIP_seed=62_ts-1651660779',
    #     'logs/bc/saved_models/gc_pick_fruit_CLIP_seed=88_ts-1651656104'
    # ]

    runs = [
        'logs/bc/saved_models/gc_pick_fruit_CLIP_best_policies'
    ]

    is_train = 'train' if (env_fn_kwargs['test_dataset'] == False) else 'test'
    best_data = {}
    all_data = collections.defaultdict(list)
    for policy_dir in runs:
        seed = -1 if is_train else int(policy_dir.split('=')[-1].split('_')[0])
        result = evaluate_grid(
            make_GC_pick_fruit_env,
            env_fn_kwargs,
            policy_dir,
            mp_ctx
        )
        best_epoch, best_acc = max(result, key=lambda x: x[1])
        best_data[seed] = (best_epoch, best_acc)
        all_data[seed].extend(result)
    
    best_success_rates = [best[1] for best in best_data.values()]
    avg_across_seeds = np.mean(best_success_rates)
    std_across_seeds = np.std(best_success_rates)
    extactor_type = None
    if env_fn_kwargs['env_kwargs']['goal_format'] == 'image':
        extactor_type = 'CNN'
    elif env_fn_kwargs['env_kwargs']['goal_format'] == 'clip':
        extactor_type = 'CLIP'
    elif env_fn_kwargs['env_kwargs']['goal_format'] == 'clip_lang':
        extactor_type = 'CLIP_Lang'

    print(f'Mean best success rate: {avg_across_seeds} | Standard dev: {std_across_seeds}')

    save_dir = 'logs/bc/results/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    stats_file = os.path.join(save_dir, f'eval_statistics_{extactor_type}_{is_train}_{time.time()}.txt')
    with open(stats_file, 'w') as f:
        f.write(f'Mean best success rate: {avg_across_seeds}\n')
        f.write(f'Std dev best success rate: {std_across_seeds}\n')
        f.write(f'=' * 10 + '\n')
        f.write('All data:\n')
        for seed, data in all_data.items():
            f.write(f'Seed: {seed} | Data: {data} | Best: {best_data[seed]}\n')


if __name__ == '__main__':
    mp_ctx = multiprocessing.get_context("forkserver")
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
    args = get_args()
    main(args, mp_ctx)
