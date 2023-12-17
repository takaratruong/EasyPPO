import time
import wandb
import os
from algs.models import Policy
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, num_envs, num_steps, gamma=0.99, lam=0.95, device='cuda'):
        self.obs_buf = torch.zeros((num_envs, num_steps, obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((num_envs, num_steps, act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((num_envs, num_steps), dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros((num_envs, num_steps), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros((num_envs, num_steps), dtype=torch.float32, device=device)

        self.val_buf = torch.zeros((num_envs, num_steps), dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros((num_envs, num_steps), dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros((num_envs, num_steps), dtype=torch.float32, device=device)

        self.device = device
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.gamma, self.lam = gamma, lam
        self.ptr, self.num_steps = 0, num_steps
        self.num_envs = num_envs

    @torch.no_grad()
    def push(self, obs, act, rew, val, logp, done):
        assert self.ptr < self.num_steps
        self.obs_buf[:, self.ptr] = torch.tensor(obs, dtype=torch.float32, device=self.device).clone()
        self.act_buf[:, self.ptr] = act.clone()
        self.rew_buf[:, self.ptr] = torch.tensor(rew, dtype=torch.float32, device=self.device).clone() if not isinstance(rew, torch.Tensor) else rew.clone()
        self.val_buf[:, self.ptr] = val.clone()
        self.logp_buf[:, self.ptr] = logp.clone()
        self.done_buf[:, self.ptr] = torch.tensor(done, dtype=torch.float32, device=self.device).clone()
        self.ptr += 1

    @torch.no_grad()
    def finish_path(self, last_val):
        assert self.ptr == self.num_steps

        # This computes the returns using GAE
        vals = torch.hstack((self.val_buf, last_val))
        last_gae = torch.zeros((self.num_envs, 1)).to(self.device)
        advs = torch.zeros_like(self.rew_buf)
        for t in reversed(range(self.num_steps)):
            delta = self.rew_buf[:, t].unsqueeze(-1) + self.gamma * vals[:, t + 1].unsqueeze(-1) * (1 - self.done_buf[:, t].unsqueeze(-1)) - vals[:, t].unsqueeze(-1)
            last_gae = delta + self.gamma * self.lam * (1 - self.done_buf[:, t].unsqueeze(-1)) * last_gae
            advs[:, t] = last_gae.squeeze()
        self.adv_buf = advs
        self.ret_buf = advs + self.val_buf

    def shuffle_and_batch(self, batch_size):
        assert self.num_envs * self.num_steps % batch_size == 0
        indices = np.arange(self.num_envs * self.num_steps)
        np.random.shuffle(indices)
        batched_idxs = np.array_split(indices, np.ceil(self.num_envs * self.num_steps / batch_size))
        return [(self.obs_buf.reshape(-1, self.obs_dim)[batch_idx].clone(), self.act_buf.reshape(-1, self.act_dim)[batch_idx].clone(),
                self.logp_buf.flatten()[batch_idx].clone(), self.ret_buf.flatten()[batch_idx].clone()) for batch_idx in batched_idxs]

    def clear(self):
        self.ptr = 0


class PPO:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        self.exp_name = config['logging']['exp_name']
        self.max_iter = config['max_iter']
        self.num_envs = config['num_envs']
        self.num_steps = config['num_steps']
        self.seed = config['seed']

        self.batch_size = config['policy']['batch_size']
        self.num_epochs = config['policy']['num_epochs']
        self.gamma = config['policy']['gamma']
        self.lam = config['policy']['lambda']

        self.state = None
        self.num_envs = env.num_envs
        
        self.obs_size = int(env.unwrapped.single_observation_space.shape[0])
        self.act_size = int(env.unwrapped.single_action_space.shape[0])
        
        self.model = Policy(self.obs_size, self.act_size, config['policy']).to(device)
        self.storage = PPOBuffer(self.obs_size, self.act_size, self.num_envs, self.num_steps, self.gamma, self.lam)


    # This function collects episode rollouts from the simulator. If you want to switch to isaac gym or pybullet, you will need to mofify this function.
    def collect_rollouts(self, num_steps):
        next_state, info = None, None

        for _ in range(num_steps):
            with torch.no_grad():
                action, log_prob, value = self.model(torch.as_tensor(self.state, dtype=torch.float32, device=device), explore=True)
            next_state, reward, term, trunc, info = self.env.step(action.cpu().numpy())
            
            self.storage.push(self.state, action, reward, value.flatten(), log_prob, term | trunc )
            self.state = next_state.copy() 
    
        # If episode terminates, gymnaisum sets the next_state as the start_state of the next episode. In these cases, the final value becomes incorrect.
        if "_final_observation" in info:
            next_state[info['_final_observation']] = info['final_observation'][info['_final_observation']][0]

        next_state = self.model.filter(torch.as_tensor(next_state, dtype=torch.float32, device=device), update=False)
        v_end = self.model.val_fn(next_state)
        self.storage.finish_path(v_end)

    def update_model(self, batch_size, num_epoch):
        val_epoch_loss, pol_epoch_loss = 0, 0
        for k in range(num_epoch):
            for states, actions, old_log_probs, returns in self.storage.shuffle_and_batch(batch_size):
                value_loss = self.model.value_loss(states, returns)
                policy_loss = self.model.policy_loss(states, actions, old_log_probs, returns)

                self.model.value_step(value_loss)
                self.model.policy_step(policy_loss)

                val_epoch_loss += value_loss.detach()
                pol_epoch_loss += policy_loss.detach()

        return val_epoch_loss / num_epoch, pol_epoch_loss / num_epoch


    def save_exp_state(self, filename, info):
        save_folder = f"results/models/{self.exp_name}"
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, filename)
        torch.save({'policy':  [self.model.state_dict(), self.model.filter.state_dict()],
                    'config': self.config,
                    'info': info}, save_path)

    def train(self):
        best_reward = -1
        info = {}

        self.state, _ = self.env.reset(seed=self.seed)
        self.state = np.array(self.state)

        for iterations in range(self.max_iter):
            print("-" * 50)
            print("iteration: ", iterations)
            iteration_start = time.time()

            self.collect_rollouts(self.num_steps)
            value_loss, policy_loss = self.update_model(self.batch_size, self.num_epochs)
            self.storage.clear()

            # Logging and Saving
            ep_lengths = self.env.length_queue if len(self.env.length_queue) > 0 else 0
            mean_ep_len = np.mean(ep_lengths)
            rewards = self.env.return_queue if len(self.env.return_queue) > 0 else 0
            mean_reward = np.mean(rewards)

            if iterations % 5 == 0:
                print(f"Reward: \u00B1 std: {mean_reward:.2f} \u00B1 {np.std(rewards):.2f}")
                print(f"Ep Len: \u00B1 std: {mean_ep_len:.2f} \u00B1 {np.std(ep_lengths):.2f}")
                wandb.log({"step": iterations, "eval/reward": mean_reward, "eval/ep_len": mean_ep_len, "train/value loss": value_loss, "train/policy loss": policy_loss})

            # Save best model
            if mean_reward >= best_reward and iterations % 25 == 0 and iterations >= 100:
                best_reward = mean_reward.copy()
                info['best_model_iter'] = iterations
                self.save_exp_state('best_model.pt', info)

            print("iteration time", np.round(time.time() - iteration_start, 3))