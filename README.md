# Naturalizing RL

**Table of Contents:**<br />
[Installation](#install)<br />
[Quick Start](#quick-start)<br />
[Advanced](#advanced)<br />
[Paper Results](#results)<br />
[TODOs](#todos)<br />


<a name="install"></a>
## Installation

In order to import and use the resources in this repository, you first need to install [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/mpiseno/habitat-lab.git).
Follow the habitat-lab instructions to also download the ReplicaCAD dataset and assets and place them in the data/ folder in your working directory.

1. Install Habitat
```bash
conda create --name habitat python=3.7
conda activate habitat
conda install habitat-sim withbullet -c conda-forge -c aihabitat-nightly
cd ~/code/
git clone -b hab_suite_dev https://github.com/mpiseno/habitat-lab.git
cd habitat-lab
pip install -r requirements.txt && python setup.py develop --all
```

**NOTE:** You must place the habitat data (e.g. the replica_cad stuff) in this repo, not in habitat-lab

**NOTE: If you are on a machine without a display (e.g. a cluster):** You will need to install habitat-sim with the headless option

```bash
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat-nightly
```

2. Install Stanford Habitat
Stanford Habitat is a collection of custom environments, measures, and sensors for habitat-lab. Note: The readme for Stanford Habitat will also tell you to install habitat-lab with a specific commit.
Ignore that and stick with Michael's forked habitat-lab version used in step 1.
```bash
cd ~/code/
git clone https://github.com/stanford-iprl-lab/stanford-habitat.git
cd stanford-habitat && pip install -e .
```

3. Install the Nat-RL package
```bash
cd ~/code/
git clone https://github.com/mpiseno/nat-rl.git
cd nat-rl && pip install -e .
```
<!-- 
4. Install Imitation
For imitation learning, we use the [imitation](https://github.com/HumanCompatibleAI/imitation) package. We use a specific commit for reproducibility.
```bash
cd ~/code/
git clone http://github.com/HumanCompatibleAI/imitation
cd imitation
git checkout -b my_specific_commit ed45793dfdd897d3ac1f3a863a8816b56d436887
pip install -e .
``` -->

<a name="quick-start"></a>
## Quick Start

### Imitiation Learning

First you need to obtain an expert dataset of image trajectories to train the policy. For expert trajectories used in our experiments, download them from [here](https://drive.google.com/file/d/139RcMbknmq4nbNSPmgWVoIHa4rvlX0d3/view?usp=sharing)
and extract the expert_trajs folder inside the data/ directory. You can also generate your own expert trajectories and clip embessings using scripts/generate_expert_trajectories.py.

To train an imitation learning agent:
```bash
python -m nat_rl.run_IL --env gc_pick_fruit --feature_extractor CNN --logdir logs/
```

The ```--feature_extractor``` argument can take either "CNN" or "CLIP". Note that CLIP embeddings are pre-computed for our IL training setup, and will be read in from data/expert_trajs/.

<a name="advanced"></a>
## Advanced

To generate a dataset using habitat-lab's data generator script (see datagen configs):
```bash
HABITAT_SIM_LOG=quiet python -m habitat.datasets.rearrange.rearrange_generator --run --config configs/pick_task/pick_fruit/pick_fruit_datagen.yaml --num-episodes 10 --out data/pick_datasets/pick_fruit_2.json.gz
```

To generate an image dataset of expert trajectories using pre-defined waypoints and IK
```bash
python scripts/generate_expert_trajs.py --env pick_fruit --generate_image_trajs
```

To view a list of all the objects/receptables available for generation:
```bash
python -m habitat.datasets.rearrange.rearrange_generator --list
```

<a name="results"></a>
## Reproducing Results

### Imitation Learning

The End-to-End baselines (using CNN on both current and goal image):

Pick Fruit:
```bash
python -m nat_rl.run_IL --env gc_pick_fruit --feature_extractor CNN --n_IL_epochs 100 --lr 0.0005 --batch_size 256 --l2_weight 0.0005 --seeds 55 7 88 62 59
```

Spatial Reasoning:

```bash
python -m nat_rl.run_IL --env gc_spatial_reasoning --feature_extractor CNN --n_IL_epochs 100 --lr 0.0005 --batch_size 256 --l2_weight 0.0005 --seeds 66 14 71 82 44
```


The CLIP-Goal baselines (Using CLIP embeddings for the goal):

Pick Fruit:
```bash
python -m nat_rl.run_IL --env gc_pick_fruit --feature_extractor CLIP --n_IL_epochs 100 --lr 0.0005 --batch_size 256 --l2_weight 0.0005 --seeds 55 7 88 62 59
```

Spatial Reasoning:
```bash
python -m nat_rl.run_IL --env gc_spatial_reasoning --feature_extractor CLIP --n_IL_epochs 100 --lr 0.0005 --batch_size 256 --l2_weight 0.0005 --seeds 66 14 71 82 44
```


### Evaluating Results

We provide an evaluation script that averages accuracies across seeds. The following command will report the average performance across best policies from each seed
inside the ```exp_logdir/saved_models/``` directory.
```bash
python scripts/evaluate_policies.py --env gc_spatial_reasoning --exp_logdir logs/bc/spatial_reasoning/CLIP_Img/ --split test --goal_type clip_img
```

```goal_tpye``` can be 'image', 'clip_img', or 'clip_lang', We expect the experiment logdir (i.e. ```exp_logdir```) to have the format.
The results from the evaluation will be placed in ```exp_logdir/results/```.

```
exp_logdir/
    results/
    saved_models/
        seed=0/
        seed=1/
        seed=2/

        ...

        seed=10/
            saved_model_epoch=0.pt
            saved_model_epoch=5.pt

            ...
```


## TODOs