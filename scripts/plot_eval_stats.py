import json
import collections
import pandas as pd
import seaborn as sns


def parse_file(path, model, Data):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = lines[4:]
        for line in lines:
            seed = int(line.split('|')[0].strip()[len('Seed: ')])
            line = line.split('|')[1].strip()[len('Data: '):]
            line = line.replace('(', '[').replace(')', ']')
            line = json.loads(line)

            line = sorted(line, key=lambda x: x[0])
            for d in line:
                epoch = d[0]
                acc = d[1]
                Data['Epoch'].append(epoch)
                Data['Val Accuracy'].append(acc)
                Data['model'].append(model)
    
    return Data


def main():
    clip_path = 'logs/bc/results/eval_statistics_CLIP_1652092107.1415632.txt'
    cnn_path = 'logs/bc/results/eval_statistics_CNN_1651990934.5464876.txt'
    D = collections.defaultdict(list)
    for path, model in [(clip_path, 'CLIP'), (cnn_path, 'CNN')]:
        D = parse_file(path, model, D)
    
    df = pd.DataFrame(D)
    plot = sns.lineplot(
        data=df, x='Epoch', y='Val Accuracy', hue='model', ci='sd'
    ).set_title('Val Task Completion Accuracy (shaded is std dev)')
    plot.get_figure().savefig('out.png')


if __name__ == '__main__':
    main()