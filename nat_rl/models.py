import numpy as np
import torch as th
import torch.nn as nn
import gym
import clip

from torchvision.transforms import Compose
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


class CLIPExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    

    Adapted from https://github.com/DLR-RM/stable-baselines3/blob/ed308a71be24036744b5ad4af61b083e4fbdf83c/stable_baselines3/common/torch_layers.py#L239
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CLIPExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}
        model, preprocess = clip.load("ViT-B/32", device='cpu') # will move everything to gpu later
        preprocess = self.edit_transforms(preprocess.transforms)

        self.model = model
        self.preprocess = preprocess
        self.freeze_model(model)

        # Compute output size
        total_concat_size = 0
        self.extractors = {}
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                with th.no_grad():
                    obs = subspace.sample()[None]
                    obs = th.as_tensor(obs.transpose(0, 3, 1, 2)).float()
                    output_size = self.model.encode_image(self.preprocess(obs)).shape[1]
                    total_concat_size += output_size
                
                self.extractors[key] = self.model.encode_image

        # total_concat_size = 0
        # for key, subspace in observation_space.spaces.items():
        #     if is_image_space(subspace):
        #         extractors[key] = CNNFeatureExtractor(subspace, features_dim=cnn_output_dim)
        #         total_concat_size += cnn_output_dim
        #     else:
        #         # The observation key is a vector, flatten it if needed
        #         extractors[key] = nn.Flatten()
        #         total_concat_size += get_flattened_obs_dim(subspace)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def edit_transforms(self, clip_transforms):
        """
        CLIP gives us their preprocessing transforms, but they assume a PIL image input. We have already normalized and converted the input to a tensor by the time the data gets here, so we need to edit the transforms
        """
        modified_transforms = Compose([
            clip_transforms[0],     # Resize 
            clip_transforms[1],     # Centercrop
            clip_transforms[4]      # Normalize
        ])
        return modified_transforms

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, observations):
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            processed_obs = self.preprocess(observations[key])
            encoded_tensor_list.append(extractor(processed_obs))

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
    
    def predict(self, observation, state=None, epiosde_start=None, deterministic=False, train_mode=None):
        assert train_mode is not None, 'train mode set to None'
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]
        
        # If we are using predict as an eval callback during training, we want to continue training after this call
        if train_mode == True:
            self.set_training_mode(True)

        return actions, state