import os
import time
import argparse
import collections
import multiprocessing

from pathlib import Path

import numpy as np

from nat_rl.utils.evaluation import evaluate_grid, SUCCESS_RATES_FNAME
from nat_rl.utils.env_utils import make_GC_pick_fruit_env, make_GC_spacial_reasoning_env


env_fns = {
    'gc_pick_fruit': make_GC_pick_fruit_env,
    'gc_spatial_reasoning': make_GC_spacial_reasoning_env
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--exp_logdir', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--goal_type', type=str, default='clip_img', choices=['image', 'clip_img', 'clip_lang', 'clip_lang_plus_init', 'clip_lang_plus_init_nn'])
    args = parser.parse_args()
    return args

def main(args, mp_ctx):
    assert args.env in env_fns

    env_fn_kwargs = {
        'test_dataset': args.split == 'test',
        'goal_type': args.goal_type,
        'env_kwargs': {
            'load_goals': True
        }
    }

    saved_models_dir = os.path.join(args.exp_logdir, 'saved_models/')
    runs = os.listdir(saved_models_dir)
    results_dir = os.path.join(args.exp_logdir, 'results/')
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    stats_file = os.path.join(results_dir, f'eval_statistics_{args.goal_type}_{args.split}_{time.time()}.txt')

    #env = env_fns[args.env](**env_fn_kwargs)

    is_train = args.split == 'train'
    best_data = {}
    all_data = collections.defaultdict(list)
    for policy_dir in runs:
        seed = int(policy_dir.split('=')[-1].split('_')[0])
        result = evaluate_grid(
            env_fns[args.env],
            env_fn_kwargs,
            os.path.join(saved_models_dir, policy_dir),
            mp_ctx
        )
        best_epoch, best_acc = max(result, key=lambda x: x[1])
        best_data[seed] = (best_epoch, best_acc)
        all_data[seed].extend(result)
    
    best_success_rates = [best[1] for best in best_data.values()]
    avg_across_seeds = np.mean(best_success_rates)
    std_across_seeds = np.std(best_success_rates)

    print(f'Mean best success rate: {avg_across_seeds} | Standard dev: {std_across_seeds}')
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
