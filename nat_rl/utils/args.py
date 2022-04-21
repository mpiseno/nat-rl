import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2_weight', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)


    parser.add_argument('--n_IL_epochs', type=int, default=10)
    parser.add_argument('--eval', action='store_true', default=False)

    args = parser.parse_args()
    return args