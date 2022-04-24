import argparse

import torch as th


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--feature_extractor', type=str, default='CNN')
    parser.add_argument('--device', type=str, default='cuda' if th.cuda.is_available() else 'cpu')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2_weight', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--n_IL_epochs', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--load_policy_path', type=str, default=None)

    parser.add_argument('--eval', type=str, default=None, choices=['train', 'test'])

    args = parser.parse_args()
    return args