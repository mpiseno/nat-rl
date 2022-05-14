import os
import json
import gzip
import random
import seaborn as sn
import pandas as pd

import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy import spatial
from sklearn.manifold import TSNE


def goal_img_sort_fn(fname):
    key = ''.join([c if c.isdigit() else '' for c in fname])
    key = int(key)
    return key


def get_episode_id_to_target_object():
    dataset_file = 'data/pick_datasets/pick_fruit/pick_fruit.json.gz'
    ep_id_to_obj = {}
    with gzip.open(dataset_file, 'rb') as f:
        episodes = json.loads(f.read())['episodes']
        for ep in episodes:
            ep_id = int(ep['episode_id'])
            obj_handle = list(ep['info']['object_labels'].keys())[0]
            obj = obj_handle.split('_')[1]
            ep_id_to_obj[ep_id] = obj
    
    return ep_id_to_obj


def generate_language(true_object):
    obj_to_adjective = {
        'apple': 'red',
        'orange': 'orange',
        'plum': 'purple',
        'banana': 'yellow'
    }
    noun1 = random.choice(['image', 'picture', 'photo'])
    verb1 = random.choice(['holding'])
    a_vs_an1 = 'A' if noun1 in ['picture', 'photo'] else 'An'
    adjective = obj_to_adjective[true_object]
    a_vs_an2 = 'a' if adjective in ['red', 'purple', 'yellow'] else 'an'
    utterance = f'A computer rendering of a robot {verb1} a {adjective} {true_object} over the table.'
    return utterance


def generate_language2(true_object):
    noun1 = random.choice(['image', 'picture', 'photo'])
    verb1 = random.choice(['holding', 'picking up', 'grabbing'])
    a_vs_an1 = 'A' if noun1 in ['picture', 'photo'] else 'An'
    a_vs_an2 = 'a' if true_object in ['plum', 'banana'] else 'an'
    utterance = f'A robot picking up {a_vs_an2} {true_object}.'
    return utterance


def generate_embeddings():
    device = 'cpu'
    model, preprocess = clip.load('ViT-B/32', device)

    expert_traj_dir = 'data/expert_trajs/pick_fruit/'
    print(f'Loading goal embeddings from expert trajectory dir: {expert_traj_dir}')

    ep_id_to_obj = get_episode_id_to_target_object()
    data = {}
    for traj_dir in os.listdir(expert_traj_dir):
        episode_id = int(traj_dir.split('=')[-1])
        clip_embeddings = np.load(os.path.join(expert_traj_dir + traj_dir, 'clip_embeddings.npy'))
        goal_embedding = clip_embeddings[-1]

        obj = ep_id_to_obj[episode_id]
        utterance = generate_language(obj)
        token = clip.tokenize([utterance])
        text_embedding = model.encode_text(token).detach().numpy()

        data[episode_id] = (goal_embedding.copy(), text_embedding[0])

    return data, ep_id_to_obj


def compute_avg_dist(img, lang, label1, label2):
    print(f'Computing {label1} <-> {label2}')
    cos_dists = []
    l2_dists = []
    for im in img:
        for u in lang:
            cos_distance = spatial.distance.cosine(im, u)
            l2_distance = np.linalg.norm(im - u)
            cos_dists.append(cos_distance)
            l2_dists.append(l2_distance)

    cos_avg = np.mean(cos_dists)
    l2_avg = np.mean(l2_dists)
    print(f'{label1} <-> {label2} | cosine: {cos_avg} l2: {l2_avg}')
    return cos_avg, l2_avg


def compute_dists(data, ep_id_to_obj):
    img_embeddings = []
    lang_embeddings = []
    labels = []
    for ep_id, (img_emb, lang_emb) in data.items():
        img_embeddings.append(img_emb)
        lang_embeddings.append(lang_emb)
        labels.append(ep_id_to_obj[ep_id])
    
    img_embeddings = np.stack(img_embeddings, axis=0)
    lang_embeddings = np.stack(lang_embeddings, axis=0)
    labels = np.array(labels)

    apple_img = img_embeddings[labels == 'apple']
    apple_lang = lang_embeddings[labels == 'apple']
    orange_img = img_embeddings[labels == 'orange']
    orange_lang = lang_embeddings[labels == 'orange']
    banana_img = img_embeddings[labels == 'banana']
    banana_lang = lang_embeddings[labels == 'banana']
    plum_img = img_embeddings[labels == 'plum']
    plum_lang = lang_embeddings[labels == 'plum']

    imgs = [
        apple_img,
        orange_img,
        banana_img,
        plum_img
    ]
    langs = [
        apple_lang,
        orange_lang,
        banana_lang,
        plum_lang
    ]

    classes = ['apple', 'orange', 'banana', 'plum']
    df_cos = pd.DataFrame(np.zeros((4, 4)), index=classes, columns=classes)
    df_l2 = pd.DataFrame(np.zeros((4, 4)), index=classes, columns=classes)
    for i, (img, label1) in enumerate(zip(imgs, classes)):
        for j, (lang, label2) in enumerate(zip(langs, classes)):
            cos, l2 = compute_avg_dist(img, lang, label1, label2)
            df_cos[label1][label2] = cos
            df_l2[label1][label2] = l2

    sn.heatmap(df_cos, annot=True, annot_kws={"size": 8}, fmt='.4f') # font size
    plt.savefig('cos_dist.png')
    plt.close()

    sn.heatmap(df_l2, annot=True, annot_kws={"size": 8}, fmt='.4f') # font size
    plt.savefig('L2_dist.png')
    plt.close()


def generate_tsne(data, ep_id_to_obj):
    img_embeddings = []
    lang_embeddings = []
    labels = []
    for ep_id, (img_emb, lang_emb) in data.items():
        img_embeddings.append(img_emb)
        lang_embeddings.append(lang_emb)
        labels.append(ep_id_to_obj[ep_id])
    
    img_embeddings = np.stack(img_embeddings, axis=0)
    lang_embeddings = np.stack(lang_embeddings, axis=0)
    labels = np.array(labels)

    all_embeddings = np.concatenate([img_embeddings, lang_embeddings], axis=0)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(all_embeddings)

    num_img = len(img_embeddings)
    apple_img = tsne[:num_img][labels == 'apple']
    apple_lang = tsne[num_img:][labels == 'apple']
    orange_img = tsne[:num_img][labels == 'orange']
    orange_lang = tsne[num_img:][labels == 'orange']
    banana_img = tsne[:num_img][labels == 'banana']
    banana_lang = tsne[num_img:][labels == 'banana']
    plum_img = tsne[:num_img][labels == 'plum']
    plum_lang = tsne[num_img:][labels == 'plum']

    plt.scatter(apple_img[:, 0], apple_img[:, 1], c='red', label='Apple (img)')
    plt.scatter(orange_img[:, 0], orange_img[:, 1], c='orange', label='Orange (img)')
    plt.scatter(banana_img[:, 0], banana_img[:, 1], c='yellow', label='Banana (img)')
    plt.scatter(plum_img[:, 0], plum_img[:, 1], c='purple', label='Plum (img)')

    plt.scatter(apple_lang[:, 0], apple_lang[:, 1], c='#F98282', label='Apple (lang)')
    plt.scatter(orange_lang[:, 0], orange_lang[:, 1], c='#F9C882', label='Orange (lang)')
    plt.scatter(banana_lang[:, 0], banana_lang[:, 1], c='#F1F982', label='Banana (lang)')
    plt.scatter(plum_lang[:, 0], plum_lang[:, 1], c='#DE82F9', label='Plum (lang)')

    plt.legend()
    plt.title('CLIP tSNE')
    plt.savefig('out.png')
    plt.close()


def compute_cosine_similarities(model, device, image, text):
    image_features = model.encode_image(image).detach().numpy()
    for t in text:
        token = clip.tokenize([t]).to(device)
        text_features = model.encode_text(token).detach().numpy()

        cos_distance = spatial.distance.cosine(image_features, text_features)
        l2_distance = np.linalg.norm(image_features - text_features)
        dot = np.dot(image_features[0], text_features[0])

        print(f'Text: {t} | Cosine distance: {cos_distance} | L2 distance: {l2_distance} | Dot: {dot}')



def compute_probabilities():
    with torch.no_grad():    
        logits_per_image, logits_per_text = model(apple, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Probs:", probs) 


def main():
    data, ep_to_obj = generate_embeddings()
    print('Generated embeddings')
    compute_dists(data, ep_to_obj)


if __name__ == '__main__':
    main()