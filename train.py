# import os
# import os.path as osp
# os.environ["MUJOCO_GL"] = "osmesa"
           
import gymnasium as gym
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
    # Disable the monitor gym since it isn't detecting the video bein created anways. there's always some weird versioning bug between gymnasium and wandb. so its just beter to do it manually.... 
    run = wandb.init(project=config['logging']['project'], config=config, name=config['logging']['exp_name'], reinit=True, monitor_gym = False, mode=config['wandb'])
    run.log_code('.') # code can be found on the wandb project page wandb/artifacts/code  

    run_id = f"/{run.id}"
    # wandb.define_metric("step", hidden=True)
    # wandb.define_metric("eval/reward", step_metric="step")
    # wandb.define_metric("eval/ep_len", step_metric="step")

    # wandb.define_metric("train/value loss", step_metric="step")
    # wandb.define_metric("train/policy loss", step_metric="step")
    
    pprint.pprint(config, sort_dicts=False)

    # Create a list of envs for training where last one is also used to record training videos to be uploaded to wandb
    envs = [lambda: gym.make(config['env']['env_id']) for _ in range(config['num_envs']-1)] + \
           [lambda: gym.wrappers.RecordVideo(gym.make(config['env']['env_id'], render_mode = 'rgb_array') , video_folder='results/videos/' + run_id,
                                                           name_prefix="rl-video", episode_trigger=lambda x: x % config['logging']['vid_rec_frq'] == 0)]

    envs = gym.vector.AsyncVectorEnv(envs)
    # envs = gym.vector.SyncVectorEnv(envs) # For debugging 

    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs, buffer_length=50)

    # Initialize RL and Train
    ppo = PPO(envs, config)
    ppo.train()

    # Close
    envs.close()
    wandb.finish()


if __name__ == '__main__':

    config = load_config()
    set_seed(config['seed'])

    train(config)
