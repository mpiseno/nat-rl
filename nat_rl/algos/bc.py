from typing import Any, Generic, Iterable, Mapping, Optional, TypeVar, Union

import torch as th
import numpy as np
import torch.utils.data as th_data

from imitation.algorithms import bc
from imitation.algorithms.bc import BehaviorCloningLossCalculator, BehaviorCloningTrainer
from imitation.data import rollout, types
from stable_baselines3.common import policies


def my_super_awesome_collate_fn(batch):
    "I stole this from https://github.com/HumanCompatibleAI/imitation/pull/415/files"

    spec_keys = ["infos", "obs", "next_obs"] if isinstance(batch[0]['obs'], dict) else ["infos"]
    batch_no_infos = [
        {k: np.array(v) for k, v in sample.items() if k not in spec_keys} for sample in batch
    ]
    result = th_data.dataloader.default_collate(batch_no_infos)
    assert isinstance(result, dict)

    # zip result["obs"] into TensorDict
    if "obs" in spec_keys:
        result["obs"] = {obs_key: np.array([sample["obs"][obs_key] for sample in batch]) \
                                        for obs_key in batch[0]["obs"]}
        result["next_obs"] = {obs_key:np.array([sample["next_obs"][obs_key] for sample in batch]) \
                                        for obs_key in batch[0]["obs"]}
    result["infos"] = [sample["infos"] for sample in batch]
    return result


def make_data_loader(transitions, batch_size, data_loader_kwargs=None):
    "modified from https://github.com/HumanCompatibleAI/imitation/blob/ed45793dfdd897d3ac1f3a863a8816b56d436887/src/imitation/algorithms/base.py"
    if batch_size <= 0:
        raise ValueError(f"batch_size={batch_size} must be positive.")

    if isinstance(transitions, Iterable):
        try:
            first_item = next(iter(transitions))
        except StopIteration:
            first_item = None
        if isinstance(first_item, types.Trajectory):
            transitions = rollout.flatten_trajectories(list(transitions))

    if isinstance(transitions, types.TransitionsMinimal):
        if len(transitions) < batch_size:
            raise ValueError(
                f"Number of transitions in `demonstrations` {len(transitions)} "
                f"is smaller than batch size {batch_size}.",
            )

        extra_kwargs = dict(shuffle=True, drop_last=True)
        if data_loader_kwargs is not None:
            extra_kwargs.update(data_loader_kwargs)
        return th_data.DataLoader(
            transitions,
            batch_size=batch_size,
            collate_fn=my_super_awesome_collate_fn,
            **extra_kwargs,
        )
    elif isinstance(transitions, Iterable):
        return _WrappedDataLoader(transitions, batch_size)
    else:
        raise TypeError(f"`demonstrations` unexpected type {type(transitions)}")


class BehaviorCloningTrainerDictObs(BehaviorCloningTrainer):
    """Functor to fit a policy to expert demonstration data."""

    loss: BehaviorCloningLossCalculator
    optimizer: th.optim.Optimizer
    policy: policies.ActorCriticPolicy

    def __call__(self, batch):
        obs = self.preprocess_obs(batch['obs'])
        acts = th.as_tensor(batch["acts"], device=self.policy.device).detach()
        training_metrics = self.loss(self.policy, obs, acts)

        self.optimizer.zero_grad()
        training_metrics.loss.backward()
        self.optimizer.step()

        return training_metrics
    
    def preprocess_img(self, obs):
        assert isinstance(obs, np.ndarray)

        obs = obs / 255
        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = th.as_tensor(obs, dtype=th.float32, device=self.policy.device).detach()
        return obs

    def preprocess_obs(self, obs):
        processed_obs = {}
        for key, val in obs.items():
            if len(val.shape) == 4:
                processed_obs[key] = self.preprocess_img(val)
            else:
                processed_obs[key] = th.as_tensor(
                    obs[key], dtype=th.float32, device=self.policy.device
                ).detach()

        return processed_obs


class BC_(bc.BC):
    '''
    The BC implementation in imiation does not support dict observations, so we have to override their functions that create their dataloaders
    '''
    def __init__(
        self,
        *,
        observation_space,
        action_space,
        policy,
        demonstrations,
        batch_size=32,
        optimizer_cls=th.optim.Adam,
        optimizer_kwargs={},
        ent_weight=1e-3,
        l2_weight=0.0,
        device="auto",
        custom_logger=None,
    ):
        super(BC_, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            policy=policy,
            demonstrations=demonstrations,
            batch_size=batch_size,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            ent_weight=ent_weight,
            l2_weight=l2_weight,
            device=device,
            custom_logger=custom_logger,
        )

        optimizer_kwargs = optimizer_kwargs or {}
        optimizer = optimizer_cls(
            self.policy.parameters(),
            **optimizer_kwargs,
        )
        loss_computer = BehaviorCloningLossCalculator(ent_weight, l2_weight)
        self.trainer = BehaviorCloningTrainerDictObs(
            loss_computer,
            optimizer,
            policy,
        )

    def set_demonstrations(self, demonstrations):
        self._demo_data_loader = make_data_loader(
            demonstrations,
            self.batch_size,
        )