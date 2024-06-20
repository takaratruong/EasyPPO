import time
import ipdb
import wandb
import os
from algs.models import Policy
import torch
import numpy as np
import collections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time


class PPOBuffer:
    def __init__(self, num_envs, horizon, obs_dim, act_dim, rew_dim, val_dim, gamma=0.99, lam=0.95, device='cuda'):
        self.obs_buf  = torch.zeros((horizon, num_envs, obs_dim)).to(device)
        self.act_buf  = torch.zeros((horizon, num_envs, act_dim)).to(device)
        self.rew_buf  = torch.zeros((horizon, num_envs, rew_dim)).to(device)
        self.val_buf  = torch.zeros((horizon, num_envs, val_dim)).to(device)
        self.stp_buf  = torch.zeros((horizon, num_envs)).to(device)
        self.logp_buf = torch.zeros((horizon, num_envs)).to(device)
        self.done_buf = torch.zeros((horizon, num_envs)).to(device)
        self.term_buf = torch.zeros((horizon, num_envs)).to(device)
        self.ret_buf  = None
        self.adv_buf  = None

        self.num_envs, self.horizon = num_envs, horizon
        self.obs_dim, self.act_dim, self.val_dim = obs_dim, act_dim, val_dim 
        self.gamma, self.lam = gamma, lam
        self.ptr = 0

    @torch.no_grad()
    def push(self, obs, stp, act, rew, val, logp, done, term):
        assert self.ptr < self.horizon
        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = act
        self.rew_buf[self.ptr]  = rew 
        self.stp_buf[self.ptr]  = stp
        self.val_buf[self.ptr]  = val
        self.logp_buf[self.ptr] = logp   
        self.done_buf[self.ptr] = done 
        self.term_buf[self.ptr] = term
        self.ptr += 1

    @torch.no_grad()    
    def finish_path(self, last_val, term_reward=None, zero_term_next_val=False):
        next_values = torch.cat((self.val_buf[1:], last_val.unsqueeze(0)), dim=0)
        
        if zero_term_next_val:
            next_values[self.term_buf.bool()] = 0
        if term_reward is not None:
            self.rew_buf[self.term_buf.bool()] = term_reward
        
        advantages = (self.rew_buf - self.val_buf).add_(next_values, alpha=self.gamma)
        for t in reversed(range(self.horizon-1)): 
            advantages[t].add_(advantages[t+1] * (1 - self.done_buf[t].unsqueeze(-1)), alpha=self.gamma * self.lam)        
        self.adv_buf = advantages
        self.ret_buf = advantages + self.val_buf 
    
    def shuffle_and_batch(self, batch_size):
        assert (self.obs_buf.shape[0] * self.obs_buf.shape[1]) % batch_size == 0, 'Batch size must evenly divide the buffer size which is num_envs * num_steps'
        assert self.adv_buf is not None and self.ret_buf is not None, "adv_buf and ret_buf must be initialized. Call finish_path first."
        
        indices = torch.randperm(self.num_envs * self.horizon) 
        batched_idxs = np.array_split(indices, np.ceil(self.num_envs * self.horizon / batch_size))
        
        return [(self.obs_buf.view(-1, self.obs_dim)[batch_idx], self.act_buf.view(-1, self.act_dim)[batch_idx],  self.logp_buf.view(-1, 1)[batch_idx], 
                 self.adv_buf.view(-1, self.val_dim)[batch_idx], self.ret_buf.view(-1, self.val_dim)[batch_idx],  self.stp_buf.view(-1)[batch_idx].long()) for batch_idx in batched_idxs]

    def clear(self):
        self.ptr = 0

class PPO:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        self.exp_name = config['exp_name']
        self.max_epochs = config['max_epochs']
        self.batch_size = config['batch_size']
        self.opt_epochs = config['opt_epochs']
        self.term_reward = config['term_reward'] 
        self.zero_term_next_val = config['zero_term_next_val'] 

        self.obs_horizon = config['env']['obs_horizon']
        self.num_steps = config['num_steps']
        self.rew_val_dim = config['rew_val_dim']
        self.state_dim = config['state_dim']

        normalize_values = config['normalize_values']
        num_envs = config['num_envs']
        goal_dim = config['goal_dim']
        explore_noise = config['explore_noise'] 
        gamma = config['gamma']
        lam = config['lambda']
        policy_lr = config['policy_lr']
        value_lr = config['value_lr']

        observation_dim = int(env.unwrapped.single_observation_space.shape[0])
        action_dim = int(env.unwrapped.single_action_space.shape[0])
        
        self.model = Policy(self.state_dim, action_dim, goal_dim, self.rew_val_dim, explore_noise, normalize_values).to(device)
        self.storage = PPOBuffer(num_envs, self.num_steps, observation_dim, action_dim, self.rew_val_dim, self.rew_val_dim, gamma, lam)

        self.optimizer = torch.optim.Adam([ {"params": self.model.policy.parameters(), "lr": policy_lr},
                                            {"params": self.model.val_fn.parameters(), "lr": value_lr }])
        self.ac_parameters = list(self.model.policy.parameters()) + list(self.model.val_fn.parameters())
        
    def collect_rollouts(self, num_steps):
        next_state, info = None, None
        state = torch.as_tensor( np.asarray(self.env.get_attr('get_obs_history')), dtype=torch.float32, device=device)
        seq_end_idx = None

        for _ in range(num_steps):
            seq_end_idx = torch.as_tensor(np.asarray(self.env.get_attr('get_seq_end_frame')), dtype=torch.long, device=device)

            with torch.no_grad():
                action, log_prob, value = self.model(state,  seq_end_idx, explore=True)
            
            next_state, reward, term, done, info = self.env.step(action.cpu().numpy())
            
            reward = np.tile(reward[:, np.newaxis], (1, 2))
        
            self.storage.push(torch.as_tensor(state), 
                    torch.as_tensor(seq_end_idx), 
                    action, 
                    # torch.as_tensor(reward).unsqueeze(-1), 
                    torch.as_tensor(reward), 
                    value, 
                    log_prob.squeeze(), 
                    torch.as_tensor(done), 
                    torch.as_tensor(term))

            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
            state = next_state.clone()
            seq_end_idx = torch.as_tensor(info['seq_idx'], dtype=torch.long, device=device)   # this is just for the last step so that the value function has the correct (next_step_seq_idx) (last step idx)

            if '_final_observation' in info and next_state is not None:
                next_state[info['_final_observation']] = torch.as_tensor(info['final_observation'][info['_final_observation']][0], dtype=torch.float32, device=next_state.device)
        
        v_end = self.model.evaluate(next_state, seq_end_idx) 
        self.storage.finish_path(v_end, self.term_reward, self.zero_term_next_val)

    def update_model(self, batch_size, num_epoch):
        val_epoch_loss, pol_epoch_loss = 0, 0
        for k in range(num_epoch):
            for obs, act, log_prob, adv, ret, seq_end_frame in self.storage.shuffle_and_batch(batch_size):
                
                value_loss = self.model.value_loss(obs, ret, seq_end_frame)
                policy_loss = self.model.policy_loss(obs, act, log_prob, adv, seq_end_frame)
                
                loss = policy_loss + value_loss
                
                self.optimizer.zero_grad()
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(self.ac_parameters, 1.0)
                self.optimizer.step()

                val_epoch_loss += value_loss.detach()
                pol_epoch_loss += policy_loss.detach()

        return val_epoch_loss / num_epoch, pol_epoch_loss / num_epoch

    def save_exp_state(self, filename, info):
        save_folder = f"results/models/{self.exp_name}"
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, filename)
        torch.save({'policy':  self.model.state_dict(),
                    'config': self.config,
                    'info': info}, save_path)

    def train(self):
        best_reward, best_track_reward = -1, -1
        lower, upper = 0, 0
        info = {}
        bin_size = 500

        self.env.reset()

        for iterations in range(int(self.max_epochs)):
            print("-" * 50)
            print("iteration: ", iterations)
            iteration_start = time.time()

            self.collect_rollouts(self.num_steps)
            
            # Update observation normalizer 
            valid_mask = torch.arange(self.obs_horizon, device=device) < self.storage.stp_buf.flatten().unsqueeze(1)
            states = self.model.process_obs(self.storage.obs_buf.clone(), norm=False)[0].reshape(-1, self.obs_horizon, self.state_dim) 
            self.model.state_normalizer.update(states[valid_mask])

            # Normalize Returns 
            if self.model.val_normalizer is not None:
                self.model.val_normalizer.update(self.storage.ret_buf.view(-1, self.rew_val_dim))
                returns = self.model.val_normalizer(self.storage.ret_buf) 
                self.storage.ret_buf = returns

            # Update model
            value_loss, policy_loss = self.update_model(self.batch_size, self.opt_epochs)
            self.storage.clear()
    
            # Logging and Saving
            ep_lengths = self.env.length_queue if self.env.length_queue else [0]
            mean_ep_len = np.mean(ep_lengths)
            rewards = self.env.return_queue if self.env.return_queue else [0]
            mean_reward = np.mean(rewards) / mean_ep_len

            if iterations % 5 == 0:
                std_reward, std_ep_len = np.std(rewards), np.std(ep_lengths)
                print(f"Reward: ± std: {mean_reward * mean_ep_len:.2f} ± {std_reward:.2f}")
                print(f"Ep Len: ± std: {mean_ep_len:.2f} ± {std_ep_len:.2f}")
                wandb.log({
                    "step": iterations, 
                    "eval/reward": mean_reward * mean_ep_len, 
                    "eval/ep_len": mean_ep_len, 
                    "eval/ave_reward_per_step": mean_reward, 
                    "train/value loss": value_loss, 
                    "train/policy loss": policy_loss,
                })

            # Save best model criteria
            if iterations % 25 == 0 and iterations >= 100 and mean_reward >= best_reward:
                best_reward = mean_reward
                info['best_model_iter'] = iterations
                self.save_exp_state('best_model.pt', info)

            # Save best model within a range
            if iterations % 50 == 0:
                if iterations % bin_size == 0:
                    lower, upper, best_track_reward = iterations, iterations + bin_size, -1
                if mean_reward > best_track_reward:
                    best_track_reward = mean_reward
                    info['best_model_iter'] = iterations
                    self.save_exp_state(f'best_model_{lower}_{upper}.pt', info)

            print(f"Iteration time: {np.round(time.time() - iteration_start, 3)}s")