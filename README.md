# Naturalizing RL
## Scratch code for "home to Habitat" project

**Table of Contents:**<br />
[Installation](#install)<br />
[Quick Start](#quick-start)<br />
[Advanced](#advanced)<br />
[TODOs](#todos)<br />


<a name="install"></a>
## Installation

In order to import and use the resources in this repository, you first need to install [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab)

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
Stanford Habitat is a collection of custom environments, measures, and sensors for habitat-lab. The readme for Stanford Habitat will also tell you to install habitat with a specific commit. Ignore that, and stick with the fork of habitat-lab we have here from step 1.
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

4. **Optional** Install Imitation
For imitation learning, we use the [imitation](https://github.com/HumanCompatibleAI/imitation) package. We use a specific commit for reproducibility.
```bash
cd ~/code/
git clone http://github.com/HumanCompatibleAI/imitation
cd imitation
git checkout -b my_specific_commit ed45793dfdd897d3ac1f3a863a8816b56d436887
pip install -e .
```

<a name="quick-start"></a>
## Quick Start

To train an imitation learning agent:
```bash
python -m nat_rl.run_IL --env gc_pick_single_object-v0
```

<a name="advanced"></a>
## Advanced

To generate a dataset using habitat-lab's data generator:
```bash
HABITAT_SIM_LOG=quiet python -m habitat.datasets.rearrange.rearrange_generator --run --config configs/pick_task/pick_single_object-datagen.yaml --num-episodes 10 --out data/pick_datasets/pick_single_object.json.gz
```

To generate an image dataset of expert trajectories using pre-defined waypoints and IK
```bash
python scripts/generate_expert_trajs.py --env pick_single_object-v0
```

## TODOs