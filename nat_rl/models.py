import torch as th
import torch.nn as nn
import gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space, preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy

from nat_rl.utils.preprocessing import maybe_transpose


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
   
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim=256):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        n_input_channels = 3 # assume rgb channels
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            obs = observation_space.sample()[None]
            obs = maybe_transpose(obs)
            n_flatten = self.cnn(th.as_tensor(obs).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").
    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.

    Adapted from https://github.com/DLR-RM/stable-baselines3/blob/ed308a71be24036744b5ad4af61b083e4fbdf83c/stable_baselines3/common/torch_layers.py#L239
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim=256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = CNNFeatureExtractor(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


class CustomMultiInputActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch = None,
        activation_fn = nn.Tanh,
        ortho_init = True,
        use_sde = False,
        log_std_init = 0.0,
        full_std =True,
        sde_net_arch = None,
        use_expln = False,
        squash_output = False,
        features_extractor_class = CustomCombinedExtractor,
        features_extractor_kwargs = None,
        normalize_images = True,
        optimizer_class = th.optim.Adam,
        optimizer_kwargs = None,
    ):
        super(CustomMultiInputActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
    
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        assert self.features_extractor is not None, "No features extractor was set"

        preprocessed_obs = maybe_transpose(obs)
        preprocessed_obs = preprocess_obs(
            preprocessed_obs, self.observation_space, normalize_images=self.normalize_images
        )
        return self.features_extractor(preprocessed_obs)