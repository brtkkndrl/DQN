import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys


class TrainingEvaluator:
    HISTORY_FILENAME = "eval_history.npz"
    GRAPH_FILENAME = "eval_graphs"
    BEST_MODEL_FILENAME = "eval_best_model.pth"

    def __init__(self, eval_env, eval_freq, runs_per_eval, rollout_freq, progress_bar):
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._runs_per_eval = runs_per_eval
        self._rollout_freq = rollout_freq

        self._best_model_reward = float("-inf")
        self._best_model_weights = None
        self._steps = 0
        self._episodes = 0

        self._episodes_in_rollout = []

        self._rollouts = []
        self._evaluations = []
        self._learning_start_time = None

        self._progress_bar = progress_bar

        self._trainer = None

    def is_worth_saving(self):
        if self._trainer is None:
            return False
        if len(self._rollouts) < 2:
            return False
        return True

    def save_history(self, path=""):
        np.savez_compressed(os.path.join(path, TrainingEvaluator.HISTORY_FILENAME), rollouts=self._rollouts, evaluations=self._evaluations)
        self.draw_graphs(show=False, save_to_file=True, path=path)

    def save_best_model_weights(self, path=""):
        if self._trainer is None:
            print("No model to save...")
            return
        
        torch.save(self._best_model_weights, os.path.join(path, TrainingEvaluator.BEST_MODEL_FILENAME))

    def draw_graphs(self, show = True, save_to_file = False, path=""):
        # training data
        train_rewards = [rollout['ep_mean_reward'] for rollout in self._rollouts]
        train_timesteps = [rollout['total_timesteps'] for rollout in self._rollouts]
        # eval data
        eval_rewards = [rollout['ep_mean_reward'] for rollout in self._evaluations]
        eval_timesteps = [rollout['total_timesteps'] for rollout in self._evaluations]

        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.plot(train_timesteps, train_rewards, linestyle='-', label="training")
        ax.plot(eval_timesteps, eval_rewards, linestyle='-', label="evaluation")

        ax.set_xlabel('Timestep')
        ax.set_ylabel('Mean episode reward')
        ax.set_title('Episode reward over timesteps')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()

        if save_to_file:
            plt.savefig(os.path.join(path, f"{TrainingEvaluator.GRAPH_FILENAME}.pdf"), format='pdf')
            plt.savefig(os.path.join(path, f"{TrainingEvaluator.GRAPH_FILENAME}.png"), format='png')
        if show:
            plt.show()

        plt.close(fig)


    def _evaluate_model(self, trainer):
        total_reward = 0.0
        total_ep_len = 0.0
        for i in range(0, self._runs_per_eval):
            obs, info = self._eval_env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                action = trainer.predict(obs)
                obs, reward, done, truncated, info = self._eval_env.step(action)
                total_reward += reward
                total_ep_len += 1

        mean_ep_reward = total_reward / self._runs_per_eval
        mean_ep_len = total_ep_len / self._runs_per_eval

        is_new_best = False

        if mean_ep_reward > self._best_model_reward:
            self._best_model_reward = mean_ep_reward
            self._best_model_weights = trainer.get_online_network_weights()
            is_new_best = True

        trainer_info = trainer.get_info()

        self._evaluations.append(
            dict(
                ep_total_len = total_ep_len,
                ep_mean_len = mean_ep_len,
                ep_mean_reward = mean_ep_reward,
                total_timesteps = trainer_info['current_timestep']
            )
        )

    def log_message(self, msg):
        print(f"----------------------------------")
        print(f"| Message from trainer:          |")
        print(f"----------------------------------")
        print(f"| {msg}")
        print(f"----------------------------------")
 
    def learn_begin(self, trainer):
        self._trainer = trainer
        self._steps = 0
        self._learning_start_time = time.time()
        print(f"----------------------------------")
        print(f"| Training started.              |")
        print(f"----------------------------------")
        return True
    
    def learn_prefill_begin(self, trainer):
        print("Prefilling buffer...")
        return True

    def learn_end(self, trainer):
        elapsed_time = time.time() - self._learning_start_time

        print(f"----------------------------------")
        print(f"| Training finished!             |")
        print(f"----------------------------------")
        print(f"|   best_eval_reward  | {self._best_model_reward :< 8.2f} |")
        print(f"|   training_time     | {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} |")
        print(f"----------------------------------")
        return True

    def _formulate_rollout(self, episodes):
        ep_total_len = sum(ep['len'] for ep in episodes)
        ep_mean_len = ep_total_len / self._rollout_freq
        ep_mean_reward = sum(ep['reward'] for ep in episodes) / self._rollout_freq

        return dict(
            ep_total_len = ep_total_len,
            ep_mean_len = ep_mean_len,
            ep_mean_reward = ep_mean_reward,
            total_timesteps = episodes[-1]['final_timestep'],
            n_episodes = episodes[-1]['n_episodes']
        )

    def _print_rollout(self, trainer, rollout, episodes):
        trainer_info = trainer.get_info()

        fps = rollout['ep_total_len'] / (episodes[-1]['end_time'] - episodes[0]['start_time'])
        elapsed_time = time.time() - self._learning_start_time
        done = rollout['total_timesteps'] / trainer_info['target_timesteps']

        print(f"----------------------------------")
        print(f"| rollout/            |          |")
        print(f"|   ep_len_mean       | {rollout['ep_mean_len'] :< 8.2f} |")
        print(f"|   ep_rew_mean       | {rollout['ep_mean_reward'] :< 8.2f} |")
        print(f"| stats/              |          |")
        print(f"|   episodes          | {rollout['n_episodes'] :< 8.0f} |")
        print(f"|   fps               | {fps :< 8.0f} |")
        print(f"|   total_timesteps   | {rollout['total_timesteps'] :< 8.0f} |")
        print(f"------------------------------------------")
        print(f"|   best_model_reward       | {self._best_model_reward :< 8.2f}   |")
        print(f"|   model_updates           | {trainer_info['model_updates'] :< 8.0f}   |")
        print(f"|   exploration_rate        | {trainer_info['exploration_rate'] :< 8.2f}   |")
        print(f"|   replay_buffer_size      | {(trainer_info['replay_buffer_size']/(1024*1024)) :< 7.2f} MB |")
        print(f"|   replay_buffer_filled    | {(trainer_info['replay_buffer_filled'] * 100.0) :< 6.2f} %  |")
        print(f"------------------------------------------")
        print(f"|   elapsed                 | {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}   |")
        print(f"|   done                    | {(done*100.0):< 7.2f}%   |")
        print(f"|   est_remaining           | {time.strftime('%H:%M:%S', time.gmtime(elapsed_time * ((1.0-done)/done)))}   |")
        print(f"------------------------------------------")

    def learn_log_episode(self, trainer, episode):
        self._episodes += 1

        self._episodes_in_rollout.append(episode)

        if self._episodes % self._rollout_freq == 0:
            rollout = self._formulate_rollout(self._episodes_in_rollout)
            self._print_rollout(trainer, rollout, self._episodes_in_rollout)
            self._rollouts.append(rollout)
            self._episodes_in_rollout = []

        if self._episodes % self._eval_freq == 0:
            self._evaluate_model(trainer)

        return True

    def train_log_loss(self, trainer, loss):
        return True

    def learn_step(self, trainer):
        self._steps += 1

        # progress bar
        if self._progress_bar:
            trainer_info = trainer.get_info()

            elapsed_time = time.time() - self._learning_start_time
            target_timesteps = trainer_info['target_timesteps']
            current_timestep = trainer_info['current_timestep']

            done = current_timestep / target_timesteps

            BAR_LENGTH = 40

            filled = int(BAR_LENGTH * done)
            bar = "=" * filled + "." * (BAR_LENGTH - filled)

            sys.stdout.write(f"\r time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} timesteps: {current_timestep}/{target_timesteps} ({(done*100):6.2f}%)[{bar}]")
            sys.stdout.flush()

        return True