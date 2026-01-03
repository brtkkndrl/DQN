import numpy as np
import torch

# TODO create an abstract ReplayBufferClass, use inheritance

class ReplayBufferDQNFramestack():
    SAVE_FILENAME = "replay_buffer.npz"

    def __init__(self, capacity, obs_space_shape, obs_space_dtype, frame_stack):
        self._replace_index = 0

        self._capacity = capacity
        self._filled_capacity = 0

        self._frame_stack_size = frame_stack

        # reserve one spot for zero frame
        self._zero_frame_index = self._capacity
        
        self._frames = np.zeros((self._capacity+1, *obs_space_shape), dtype=obs_space_dtype)
        self._actions = np.zeros(self._capacity, np.uint8)
        self._rewards = np.zeros(self._capacity, np.float32)
        self._dones = np.zeros(self._capacity, bool)
        self._is_valid = np.zeros(self._capacity, bool)

    def get_allocated_size(self):
        """
            Returns allocated memmory size in bytes.
        """
        bytes = self._frames.nbytes + self._actions.nbytes + self._rewards.nbytes+self._dones.nbytes+self._is_valid.nbytes
        return bytes
    
    def get_filled_ratio(self):
        return self._filled_capacity / self._capacity

    def fill_frames(self, frame):
        """
            Fill last frames to complete observation.
        """
        framestack_ids = np.full(
            self._frame_stack_size-1,
            self._zero_frame_index,
            dtype=np.int32
        )

        last_id = (self._replace_index-1)%self._capacity

        framestack_ids[0] = last_id

        for i in range(1, self._frame_stack_size-1):
            prev_id = (framestack_ids[i-1] - 1) % self._capacity
            # cannot use invalid frames or past episode boundries
            valid_prev = (framestack_ids[i-1] != self._zero_frame_index) & (~self._dones[prev_id]) & self._is_valid[prev_id]
            # replace non valid with zeros
            framestack_ids[i] = np.where(valid_prev, prev_id, self._zero_frame_index)

        frames = self._frames[framestack_ids] # get frames from ids
        frames = np.concatenate((frame[None, :, :], frames), axis=0) # add current frame to the from
        frames = frames[::-1] # reverse it so current frame is last
        frames = frames.copy() # copy to avoid negative stride

        return frames 

    
    def sample_tensors(self, batch_size, device):
        valid_ids = np.nonzero(self._is_valid)[0]

        ids = np.random.choice(valid_ids, batch_size, replace=False)
        next_ids = (ids + 1) % self._capacity

        framestack_ids = np.full(
            (self._frame_stack_size+1, batch_size),
            self._zero_frame_index,
            dtype=np.int32
        )
        framestack_ids[0] = next_ids
        framestack_ids[1] = ids

        for i in range(2, self._frame_stack_size+1):
            prev_ids = (framestack_ids[i-1] - 1) % self._capacity
            # cannot use invalid frames or past episode boundries
            valid_prev = (framestack_ids[i-1] != self._zero_frame_index) & (~self._dones[prev_ids]) & self._is_valid[prev_ids]
            # replace non valid with zeros
            framestack_ids[i] = np.where(valid_prev, prev_ids, self._zero_frame_index)

        obs_ids = (framestack_ids[1:self._frame_stack_size+1][::-1]).T  # frame stack until obs
        next_obs_ids = (framestack_ids[0:self._frame_stack_size][::-1]).T    # frame stack unit next obs

        obs = torch.from_numpy(self._frames[obs_ids]).float().to(device)
        actions = torch.from_numpy(self._actions[ids]).long().unsqueeze(-1).to(device)
        rewards = torch.from_numpy(self._rewards[ids]).float().unsqueeze(-1).to(device)
        dones = torch.from_numpy(self._dones[ids]).float().unsqueeze(-1).to(device)
        next_obs = torch.from_numpy(self._frames[next_obs_ids]).float().to(device)

        return obs, actions, rewards, dones, next_obs
    
    def update(self, obs, action, reward, done):
        # store observation
        self._frames[self._replace_index] = obs
        self._actions[self._replace_index] = action
        self._rewards[self._replace_index] = reward
        self._dones[self._replace_index] = done

        # update validity
        self._is_valid[self._replace_index] = False # this becomes invalid as it has no next_obs yet
        self._is_valid[(self._replace_index-1) % self._capacity] = True # previous becomes valid

        # update indexing
        self._replace_index = (self._replace_index+1) % self._capacity
        self._filled_capacity = min(self._filled_capacity+1, self._capacity)

    def save_to_file(self, path):
        np.savez(f"{path}{ReplayBufferDQNFramestack.SAVE_FILENAME}",
            replace_index = self._replace_index,
            capacity = self._capacity,
            frame_stack_size = self._frame_stack_size,
            zero_frame_index = self._zero_frame_index,
            frames = self._frames,
            actions = self._actions,
            rewards = self._rewards,
            dones = self._dones,
            is_valid = self._is_valid
        )


class ReplayBufferDQNBasic():
    def __init__(self, capacity, obs_space_shape, obs_space_dtype,):
        self._replace_index = 0
        self._capacity = capacity
        self._filled_capacity = 0

        self._obs = np.zeros((self._capacity, *obs_space_shape), dtype=obs_space_dtype)
        self._actions = np.zeros(self._capacity, np.uint8)
        self._rewards = np.zeros(self._capacity, np.float32)
        self._dones = np.zeros(self._capacity, bool)
        self._is_valid = np.zeros(self._capacity, bool)
    
    def sample_tensors(self, batch_size, device):
        valid_ids = np.nonzero(self._is_valid)[0]
        ids = np.random.choice(valid_ids, batch_size, replace=False)

        next_ids = (ids + 1) % self._capacity

        obs = torch.from_numpy(self._obs[ids]).float().to(device)
        actions = torch.from_numpy(self._actions[ids]).long().unsqueeze(-1).to(device)
        rewards = torch.from_numpy(self._rewards[ids]).float().unsqueeze(-1).to(device)
        dones = torch.from_numpy(self._dones[ids]).float().unsqueeze(-1).to(device)
        next_obs = torch.from_numpy(self._obs[next_ids]).float().to(device)

        return obs, actions, rewards, dones, next_obs
    
    
    def get_allocated_size(self):
        """
            Returns allocated memmory size in bytes.
        """
        bytes = self._obs.nbytes + self._actions.nbytes + self._rewards.nbytes+self._dones.nbytes+self._is_valid.nbytes
        return bytes
    
    def get_filled_ratio(self):
        return self._filled_capacity / self._capacity

    def update(self, obs, action, reward, done):
        # store observation
        self._obs[self._replace_index] = obs
        self._actions[self._replace_index] = action
        self._rewards[self._replace_index] = reward
        self._dones[self._replace_index] = done

        # update validity
        self._is_valid[self._replace_index] = False # this becomes invalid as it has no next_obs yet
        self._is_valid[(self._replace_index-1) % self._capacity] = True # previous becomes valid

        # update indexing
        self._replace_index = (self._replace_index+1) % self._capacity
        self._filled_capacity = min(self._filled_capacity+1, self._capacity)