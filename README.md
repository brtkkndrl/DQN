# Deep Q-Networks

This project implements a framework for training Deep Q-Networks with enhancements such as **Double DQN**, **Dueling DQN**, and **Lazy Frame Stacking**. The core Deep Q-Learning implementation is based on the paper ["Playing Atari with Deep Reinforcement Learning"](http://arxiv.org/abs/1312.5602) by Mnih et al. (2013).

[Example usage](src/dqn/examples/cartpole.py)

### Example training
```
----------------------------------
| rollout/            |          |
|   ep_len_mean       |  153.75  |
|   ep_rew_mean       |  153.75  |
| stats/              |          |
|   episodes          |  132     |
|   fps               |  319     |
|   total_timesteps   |  6081    |
------------------------------------------
|   best_model_reward       |  279.60    |
|   model_updates           |  5081      |
|   exploration_rate        |  0.01      |
|   replay_buffer_size      |  0.22   MB |
|   replay_buffer_filled    |  61.32 %  |
------------------------------------------
|   elapsed                 | 00:00:18   |
|   done                    |  60.81 %   |
|   est_remaining           | 00:00:11   |
------------------------------------------
```

### Install package
```
pip install git https://github.com/brtkkndrl/DQN.git
```

### Example training script:
```
dqn_example_cartpole
```

### Development
```
python -m venv venv
source venv/bin/activate
pip install -e .[examples]
```
