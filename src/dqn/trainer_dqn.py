import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim
import numpy as np
import time
import random
import json
import zipfile
import io

from dataclasses import dataclass

from .replay_buffer import ReplayBufferDQNFramestack, ReplayBufferDQNBasic

class ModelDQNDueling(nn.Module):
    def __init__(self, feature_extractor, actions_dim):
        super().__init__()

        if not hasattr(feature_extractor, "features_dim"):
            raise ValueError("Missing attribute features_dim on the feature extractor.")

        self.feature_extractor = feature_extractor
        self.value_head = nn.Linear(self.feature_extractor.features_dim, 1)
        self.advantage_head = nn.Linear(self.feature_extractor.features_dim, actions_dim)

    def forward(self, x):
        #x /= 255.0 # normalize
        features = self.feature_extractor(x)
        advantages = self.advantage_head(features)
        q_values = self.value_head(features) + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
    def predict(self, obs):
        with torch.no_grad():
            q_values = self.forward(obs)
            return torch.argmax(q_values, dim=1).item()

class ModelDQN(nn.Module):
    def __init__(self, feature_extractor, actions_dim):
        super().__init__()

        if not hasattr(feature_extractor, "features_dim"):
            raise ValueError("Missing attribute features_dim on the feature extractor.")

        self.feature_extractor = feature_extractor
        self.q_head = nn.Linear(self.feature_extractor.features_dim, actions_dim)

    def forward(self, x):
        #x /= 255.0 # normalize
        features = self.feature_extractor(x)
        q_values = self.q_head(features)
        return q_values
    
    def predict(self, obs):
        with torch.no_grad():
            q_values = self.forward(obs)
            return torch.argmax(q_values, dim=1).item()

@dataclass
class TrainerDQNHyperparams():
    learning_rate: float =1e-4
    exploration_rate_initial: float =0.1
    exploration_rate_final: float =0.01
    exploration_fraction: float =0.1 # per training timesteps
    gamma: float =0.99
    replay_buffer_size: int =50_000
    batch_size: int =32
    train_freq: int =4
    learning_starts: int = 10_000
    target_update_interval: int =256
    frame_stack: int = 4
    double_dqn: bool =False
    dueling_dqn: bool = False
    
    def to_json_str(self):
        return json.dumps(self.__dict__, indent=2)
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_json_fs(cls, filestream):
        data = json.load(filestream)
        return cls(**data)
    
    @classmethod
    def from_file(cls, filepath):
        with open(filepath, 'r') as f:
            return cls.from_json_fs(f)

class TrainerDQN:
    def __init__(self, obs_space_shape, obs_space_dtype, action_space_dim,
                  hyperparameters, model_class, model_kwargs, use_cuda_device, mode = "train", callback = None):
        if not isinstance(hyperparameters, TrainerDQNHyperparams):
            raise TypeError(f"Invalid type for hyperparameters, please use: {TrainerDQNHyperparams.__name__}.")

        if not (mode == "train" or mode == "eval"):
            raise ValueError(f"Invalid mode ({mode}), valid modes: [train, eval].")

        self._mode = mode
        self.hyperparameters = hyperparameters
        self._callback = callback

        # setup models
        q_network_class = ModelDQNDueling if self.hyperparameters.dueling_dqn else ModelDQN
        obs_shape = (self.hyperparameters.frame_stack, *obs_space_shape)
        self._online_network = q_network_class(
            feature_extractor = model_class(obs_shape, **model_kwargs), actions_dim = action_space_dim
        )
        self._target_network = q_network_class(
            feature_extractor = model_class(obs_shape, **model_kwargs), actions_dim = action_space_dim
        )
        self._update_target_network()

        self._device = torch.device("cuda" if (use_cuda_device and torch.cuda.is_available()) else "cpu")
        self._online_network.to(self._device)
        self._target_network.to(self._device)

        # setup optimizer
        self._optimizer = optim.Adam(self._online_network.parameters(), lr=self.hyperparameters.learning_rate)

        # remember model data
        self._model_metadata = dict(
            model_class = model_class.__name__,
            model_kwargs = model_kwargs,
        )

        if self._mode == "train":
            # replay buffer
            if self.hyperparameters.frame_stack > 1:
                self._replay_buffer = ReplayBufferDQNFramestack(capacity = self.hyperparameters.replay_buffer_size,
                                                            obs_space_shape = obs_space_shape,
                                                            obs_space_dtype = obs_space_dtype,
                                                            frame_stack = self.hyperparameters.frame_stack)
            else:
                self._replay_buffer = ReplayBufferDQNBasic(capacity=self.hyperparameters.replay_buffer_size,
                                                           obs_space_shape = obs_space_shape,
                                                           obs_space_dtype = obs_space_dtype)

        # training progress parameters
        self._current_timestep = 0
        self._target_timesteps = None # defined when learn is called
        self._model_updates = 0 # how many times was online policy updated in trainings
        
        self._env = None # set when learn is called
        
    #------------------------------------
    #       public functions
    #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    def load_state(self, weights_online, weights_target,
                   current_timestep, model_updates, target_timesteps):
        # load progress
        self._current_timestep = current_timestep
        self._model_updates = model_updates
        self._target_timesteps = target_timesteps

        # load model weights
        self._online_network.load_state_dict(weights_online)
        self._target_network.load_state_dict(weights_target)

    def load_weights(self, filepath):
        weights = torch.load(filepath, map_location=torch.device('cpu'))
        self._online_network.load_state_dict(weights)
        self._update_target_network()

        self._online_network.to(self._device)
        self._target_network.to(self._device)


    @classmethod
    def load_from_file(cls, obs_space_shape, obs_space_dtype, action_space_dim, filepath, model_class, use_cuda_device,
                       new_hyperparams = None, mode="train", callback=None):
        with zipfile.ZipFile(f"{filepath}", "r") as zip_file:
            # load hyperparameters
            with zip_file.open("hyperparameters.json") as f:
                hyperparameters = TrainerDQNHyperparams.from_json_fs(f)

            # load progress data
            with zip_file.open("progress.json", "r") as f:
                progress_data = json.load(f)

            # load model metadata
            with zip_file.open("model_metadata.json", "r") as f:
                model_metadata = json.load(f)

            # load model weights
            with zip_file.open("weights_online.pth") as f:
                weights_online = torch.load(io.BytesIO(f.read()), map_location=torch.device('cpu'))
            #
            with zip_file.open("weights_target.pth") as f:
                weights_target = torch.load(io.BytesIO(f.read()), map_location=torch.device('cpu'))

        # check model class name match
        if model_class.__name__ != model_metadata['model_class']:
            raise TypeError(f"Model class mismatch! parameter: {model_class.__name__} vs file: {model_metadata['model_class']}")
        
        if new_hyperparams is not None:
            # check if new_hyperparams are compatible the previous
            if new_hyperparams.dueling_dqn != hyperparameters.dueling_dqn:
                raise ValueError("Hyperparams not compatible with old training, dueling mismatch.")
            hyperparameters = new_hyperparams

        dqn = cls(obs_space_shape = obs_space_shape,
                  obs_space_dtype = obs_space_dtype,
                  action_space_dim = action_space_dim,
                  hyperparameters= hyperparameters,
                  model_class=model_class, model_kwargs=model_metadata['model_kwargs'],
                  use_cuda_device = use_cuda_device, mode = mode, callback=callback)
        
        dqn.load_state(weights_online=weights_online,
                       weights_target=weights_target,
                       current_timestep = progress_data["current_timestep"],
                       model_updates = progress_data["model_updates"],
                       target_timesteps = progress_data["target_timesteps"])

        return dqn

    def save(self, filepath):
        with zipfile.ZipFile(f"{filepath}", "w") as zip_file:
            # save hyperparameters
            zip_file.writestr("hyperparameters.json", self.hyperparameters.to_json_str())

            # save progress data
            progress_data = dict(
                current_timestep = self._current_timestep,
                model_updates = self._model_updates,
                target_timesteps = self._target_timesteps
            )
            zip_file.writestr("progress.json", json.dumps(progress_data, indent=2))

            # save model metadata
            zip_file.writestr("model_metadata.json", json.dumps(self._model_metadata, indent=2))

            # save model weights
            buffer = io.BytesIO()
            torch.save(self._online_network.state_dict(), buffer)
            buffer.seek(0)
            zip_file.writestr("weights_online.pth", buffer.read())
            #
            buffer = io.BytesIO()
            torch.save(self._target_network.state_dict(), buffer)
            buffer.seek(0)
            zip_file.writestr("weights_target.pth", buffer.read())

    def get_info(self):
        return {
            'current_timestep'     : self._current_timestep,
            'target_timesteps'     : self._target_timesteps,
            'model_updates'        : self._model_updates,
            'exploration_rate'     : self._get_exploration_rate(self._current_timestep, self._target_timesteps),
            'replay_buffer_size'   : self._replay_buffer.get_allocated_size(),
            'replay_buffer_filled' : self._replay_buffer.get_filled_ratio()
        }

    def get_online_network_weights(self):
        return self._online_network.state_dict()
    

    def predict(self, obs):
        return self._online_network.predict(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device))

    def learn(self, env, target_timesteps):
        if self._mode != "train":
            raise ValueError("Cannot use learn() in evaluation mode.")

        # learning begin callback
        if self._callback:
            self._callback.learn_begin(self)

        self._env = env
        self._model_updates = 0
        self._current_timestep = 0
        self._target_timesteps = target_timesteps

        # initialize replay memmory
        self._prefill_replay_buffer(N=50)

        # reset env
        obs, info = self._env.reset()
        # reset episode
        episode = self._get_blank_episode()
        n_episodes = 0

        while self._current_timestep < self._target_timesteps:
            # select action
            exploration_rate = self._get_exploration_rate(self._current_timestep, self._target_timesteps)
            if isinstance(self._replay_buffer, ReplayBufferDQNFramestack):
                action = self._select_action(self._replay_buffer.fill_frames(obs), exploration_rate)
            else:
                action = self._select_action(obs, exploration_rate)
            # execute action
            next_obs, reward, done, truncated, info = self._env.step(action)
            # store transition in replay buffer
            self._replay_buffer.update(obs, action, reward, (done or truncated))

            episode['len'] += 1
            episode['reward'] += reward

            if done or truncated:
                # reset enviroment
                next_obs, info = self._env.reset()

                # finalize episode
                episode['truncated'] = truncated
                episode['final_timestep'] = self._current_timestep
                episode['end_time'] =  time.perf_counter()
                n_episodes += 1
                episode['n_episodes'] = n_episodes

                # episode callback
                if self._callback:
                    self._callback.learn_log_episode(self, episode)

                # reset episode
                episode = self._get_blank_episode()

            obs = next_obs

            self._current_timestep += 1

            if self._current_timestep > self.hyperparameters.learning_starts and self._current_timestep % self.hyperparameters.train_freq == 0:
                self._train()

            if self._callback:
                if self._callback.learn_step(self) == False:
                    break # stop training


        # learning end callback
        if self._callback:
            self._callback.learn_end(self)

    #------------------------------------
    #       private functions
    #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    def _update_target_network(self):
      self._target_network.load_state_dict(self._online_network.state_dict())

    def _select_action(self, obs, exploration_rate):
        if random.random() < exploration_rate: # explore
            # select a random action
            return self._env.action_space.sample()
        else: # exploit
            # select action that maximazies Q at given state
            return self.predict(obs)

    def _get_exploration_rate(self, timestep, total_timesteps):
        ep_initial = self.hyperparameters.exploration_rate_initial
        ep_fraction = self.hyperparameters.exploration_fraction
        ep_final = self.hyperparameters.exploration_rate_final
        rate = (timestep-self.hyperparameters.learning_starts) / ((total_timesteps-self.hyperparameters.learning_starts)*ep_fraction)
        return  ep_initial + max(0.0, min(1.0, rate))*(ep_final-ep_initial)

    def _prefill_replay_buffer(self, N=50):
        obs, info = self._env.reset()

        # initialize replay memmory
        for _ in range(0, N):
            # select random action
            action = self._env.action_space.sample()
            # execute action
            next_obs, reward, done, truncated, info = self._env.step(action)
            # store transition in replay buffer
            self._replay_buffer.update(obs, action, reward, (done or truncated))
            if done or truncated:
                next_obs, info = self._env.reset()
            obs = next_obs

    def _get_blank_episode(self):
        return dict(
            n_episodes = 0,
            len = 0,
            reward = 0,
            final_timestep = 0,
            start_time = time.perf_counter(),
            end_time = None,
            truncated = False
        )

    def _train(self):
        # sample random batch from replay buffer
        obs, actions, rewards, dones, next_obs = self._replay_buffer.sample_tensors(self.hyperparameters.batch_size, self._device)

        # compute targets
        q_values_targets = None
        with torch.no_grad():
            q_values_next_obs = None

            if self.hyperparameters.double_dqn:
                max_actions_next = torch.argmax(self._online_network.forward(next_obs), dim=1).unsqueeze(-1)
                q_values_next_obs = torch.gather(self._target_network.forward(next_obs), dim=1, index=max_actions_next)
            else:
                q_values_next_obs, _ = self._target_network.forward(next_obs).max(dim=1, keepdim=True)

            q_values_targets = rewards + (1 - dones) * self.hyperparameters.gamma * q_values_next_obs

        # compute predictions
        q_values_predictions = torch.gather(self._online_network.forward(obs), dim=1, index=actions)
        # compute loss
        loss = nn_functional.smooth_l1_loss(q_values_predictions, q_values_targets)
        # perform gradient descend
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # update target network
        self._model_updates += 1
        if self._model_updates % self.hyperparameters.target_update_interval == 0:
            self._update_target_network()

        # training loss callback
        if self._callback:
            self._callback.train_log_loss(self, loss.item())
