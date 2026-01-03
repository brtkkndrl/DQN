from dqn import TrainingEvaluator, TrainerDQN, TrainerDQNHyperparams
import gymnasium as gym
import torch.nn as nn

class CartPoleNN(nn.Module):
    def __init__(self, observation_shape, features_dim=64):
        super(CartPoleNN, self).__init__()
        _, n_inputs = observation_shape

        self.features_dim = features_dim

        # Simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


def main():
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    callback = TrainingEvaluator(eval_env=eval_env, eval_freq=4, runs_per_eval=5, rollout_freq = 4, progress_bar=False)

    dqn = TrainerDQN(
        hyperparameters = TrainerDQNHyperparams(
            learning_rate=2e-4,
            exploration_fraction=0.1,
            exploration_rate_initial=0.9,
            exploration_rate_final=0.01,
            replay_buffer_size = 10_000,
            learning_starts = 1000,
            gamma = 0.99,
            batch_size = 32,
            target_update_interval=512,
            train_freq = 1,
            double_dqn = True,
            dueling_dqn=True,
            frame_stack=1
        ),
        model_class=CartPoleNN,
        model_kwargs={"features_dim": 90},
        obs_space_dtype=env.observation_space.dtype,
        obs_space_shape=env.observation_space.shape,
        action_space_dim=env.action_space.n,
        use_cuda_device=True,
        callback = callback
    )

    dqn.learn(env=env, target_timesteps=10_000)

    callback.draw_graphs()

if __name__ == "__main__":
    main()
