import gym
import env  # MUST INCLUDE (otherwise gym can't find the environment)
import wandb
from configs.config_loader import load_config
import numpy as np
from algs.ppo import PPO
import torch
import random
import pprint

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def train(config):
    run = wandb.init(project=config['logging']['project'], config=config, name=config['exp_name'], reinit=True, monitor_gym = True, mode=config['wandb'])
    run.log_code('.')

    run_id = f"/{run.id}"
    wandb.define_metric("step", hidden=True)
    wandb.define_metric("eval/reward", step_metric="step")
    wandb.define_metric("eval/amp-reward", step_metric="step")
    wandb.define_metric("eval/ep_len", step_metric="step")

    wandb.define_metric("train/value loss", step_metric="step")
    wandb.define_metric("train/policy loss", step_metric="step")

    pprint.pprint(config, sort_dicts=False)

    # Create a list of envs for training where the last one is also used to record videos
    envs = [lambda config=config: gym.make(config['env']['env_id'], config=config, new_step_api=True) for _ in range(config['num_envs'] - 1)] + \
           [lambda config=config: gym.wrappers.RecordVideo(gym.make(config['env']['env_id'], config=config, new_step_api=True, render_mode='rgb_array'), video_folder='results/videos/' + run_id,
                                                           name_prefix="rl-video", episode_trigger=lambda x: x % config['logging']['vid_rec_frq'] == 0, new_step_api=True)]

    # Vectorize environments w/ multi-processing
    envs = gym.vector.AsyncVectorEnv(envs, new_step_api=True, shared_memory=True)
    # envs = gym.vector.SyncVectorEnv(envs, new_step_api=True), print("USING SYNC VECTOR ")

    # Wrap to record ep rewards and ep lengths
    envs = gym.wrappers.RecordEpisodeStatistics(envs, new_step_api=True, deque_size=50)

    # Initialize RL and Train
    ppo = PPO(envs, config)
    ppo.train()

    # Close
    envs.close()
    wandb.finish()


if __name__ == '__main__':

    # Either pass the path into load_config('configs/sub2_walk.py') or put as flag in command line: python run.py -config configs/sub2_walk.py
    config = load_config()
    set_seed(1)

    train(config)
