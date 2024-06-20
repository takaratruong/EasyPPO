import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiagonalPopArt(torch.nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor, bias: torch.Tensor, momentum:float=0.1):
        super().__init__()
        self.epsilon = 1e-5

        self.momentum = momentum
        self.register_buffer("m", torch.zeros((dim,), dtype=torch.float64))
        self.register_buffer("v", torch.full((dim,), self.epsilon, dtype=torch.float64))
        self.register_buffer("debias", torch.zeros(1, dtype=torch.float64))

        self.weight = weight
        self.bias = bias

    def forward(self, x, unnorm=False):
        debias = self.debias.clip(min=self.epsilon)   # type: ignore
        mean = self.m/debias
        var = (self.v - self.m.square()).div_(debias) # type: ignore
        if unnorm:
            return (mean + torch.sqrt(var) * x).to(x.dtype)

        return ((x - mean) * torch.rsqrt(var)).to(x.dtype)

    @torch.no_grad()
    def update(self, x):
        # ART
        running_m = torch.mean(x, dim=0)
        running_v = torch.mean(x.square(), dim=0)
        new_m = self.m.mul(1-self.momentum).add_(running_m, alpha=self.momentum) # type: ignore
        new_v = self.v.mul(1-self.momentum).add_(running_v, alpha=self.momentum) # type: ignore
        
        # POP 
        std = (self.v - self.m.square()).sqrt_() # type: ignore
        new_std_inv = (new_v - new_m.square()).rsqrt_()

        scale = std.mul_(new_std_inv)
        shift = (self.m - new_m).mul_(new_std_inv)

        self.bias.data.mul_(scale).add_(shift)
        self.weight.data.mul_(scale.unsqueeze_(-1))

        self.debias.data.mul_(1-self.momentum).add_(1.0*self.momentum) # type: ignore
        self.m.data.copy_(new_m) # type: ignore
        self.v.data.copy_(new_v) # type: ignore



class Critic(torch.nn.Module):
        def __init__(self, state_dim, goal_dim, value_dim=1, latent_dim=256):
            super().__init__()
            self.rnn = torch.nn.GRU(state_dim, latent_dim, batch_first=True)

            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(latent_dim+goal_dim, 1024),
                torch.nn.ReLU6(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU6(),
                torch.nn.Linear(512, value_dim)
            )
            
            for n, p in self.mlp.named_parameters():
                if "bias" in n:
                    torch.nn.init.constant_(p, 0.)
                elif "weight" in n:
                    torch.nn.init.uniform_(p, -0.0001, 0.0001)
            self.all_inst = torch.arange(0)

        def forward(self, state, seq_end_frame, goal=None):
            n_inst = state.size(0)

            if n_inst > self.all_inst.size(0):
                self.all_inst = torch.arange(n_inst, dtype=seq_end_frame.dtype, device=seq_end_frame.device)
            state, _ = self.rnn(state)

            # pull the last one out (recall that this is a GRU, so the last one contains all the needed information)
            state = state[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=state.size(1)-1))]
            
            if goal is not None:
                state = torch.cat((state, goal), -1)

            return self.mlp(state)

class Actor(torch.nn.Module):
    def __init__(self, state_dim, goal_dim, act_dim, latent_dim=256, explore_noise=None):
        super().__init__()
        
        self.rnn = torch.nn.GRU(state_dim, latent_dim, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim+goal_dim, 512),
            torch.nn.ReLU6(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU6()
        )

        self.mu = torch.nn.Linear(512, act_dim)
        self.log_sigma = torch.nn.Linear(512, act_dim)
        with torch.no_grad():
            torch.nn.init.constant_(self.log_sigma.bias, -3)
            torch.nn.init.uniform_(self.log_sigma.weight, -0.0001, 0.0001)
            self.all_inst = torch.arange(0)
        
        self.explore_noise = explore_noise
        self.all_inst = torch.arange(0)

    def forward(self, state, seq_end_frame, goal=None):
        if self.rnn is None:
            state = state.view(state.size(0), -1)
        else:
            n_inst = state.size(0)
            if n_inst > self.all_inst.size(0):
                self.all_inst = torch.arange(n_inst, dtype=seq_end_frame.dtype, device=seq_end_frame.device)

            # Embed the sequence of states, then pull out the last state of each sequence (remove the padding)    
            state, _ = self.rnn(state)
            state = state[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=state.size(1)-1))]

        if goal is not None:
            state = torch.cat((state, goal), -1)
        latent = self.mlp(state)

        mu = self.mu(latent)
        
        if self.explore_noise is None:
            sigma = torch.exp(self.log_sigma(latent)) + 1e-8
        else:
            sigma = self.explore_noise
        
        return torch.distributions.normal.Normal(mu, sigma)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, value_dim, explore_noise=None, noramlize_values=False):
        super(Policy, self).__init__()
        
        assert value_dim <= 1 or noramlize_values, "Normalization must be enabled when value_dim > 1"
        
        self.explore_noise = explore_noise        
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        
        self.policy = Actor(self.state_dim, self.goal_dim, action_dim, explore_noise=explore_noise) 
        self.val_fn = Critic(self.state_dim, self.goal_dim, value_dim)

        self.state_normalizer = RunningMeanStd(state_dim, 10.0) 
        self.val_normalizer = DiagonalPopArt(value_dim, self.val_fn.mlp[-1].weight, self.val_fn.mlp[-1].bias) if noramlize_values else None # type: ignore
        
    def process_obs(self, obs, norm=True):
        state, goal = obs, None 
        if self.goal_dim > 0: 
            state = obs[...,:-self.goal_dim]
            goal = obs[...,-self.goal_dim:] 
        state = state.view(*state.shape[:-1], -1, self.state_dim)
        return self.state_normalizer(state) if norm else state, goal

    def forward(self, obs, seq_end_frame, explore=True):
        state, goals = self.process_obs(obs, norm=False)
        state = self.state_normalizer(state)    

        val = self.val_fn(state, seq_end_frame, goals)
        if self.val_normalizer is not None:                     
            val = self.val_normalizer(val, unnorm=True) 
        
        pi = self.policy(state, seq_end_frame, goals)
        action = pi.sample()
        log_prob = pi.log_prob(action).sum(-1)

        if explore == False:
            action = pi.loc
        
        return action, log_prob, val 

    def value_loss(self, obs, returns, seq_end_frame):

        # self.val_normalizer.update(returns) 
        # returns = self.val_normalizer(returns)
        
        states, goals = self.process_obs(obs) 
        curr_values = self.val_fn(states, seq_end_frame, goals)
        assert curr_values.shape == returns.shape, f"curr_values: {curr_values.shape}, returns: {returns.shape}"
        
        critic_loss = .5* (curr_values - returns).square().mean() 

        return critic_loss

    def policy_loss(self, obs, actions, old_log_probs, advantages, seq_end_frame=None):
        states, goals = self.process_obs(obs)

        sigma, mu = torch.std_mean(advantages, dim=0, unbiased=True)
        advantages = (advantages - mu) / (sigma + 1e-5) 
        
        pi = self.policy(states, seq_end_frame, goals) 
        log_probs = pi.log_prob(actions).sum(-1, keepdim=True)

        assert log_probs.shape == old_log_probs.shape, f"log_probs: {log_probs.shape}, old_log_probs: {old_log_probs.shape}"
        ratio = torch.exp(log_probs - old_log_probs) 

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1-.2, 1+.2) * advantages
        
        actor_loss = -torch.min(surr1, surr2 ).sum(-1).mean() # + .01* (mean_actions ** 2).mean() e.g., save the mean action in the buffer and then use it here 

        return actor_loss

    def evaluate(self, obs, seq_end_frame=None):
        state, goal = self.process_obs(obs)
        val = self.val_fn(state, seq_end_frame, goal)   
        if self.val_normalizer is not None:                     
            val = self.val_normalizer(val, unnorm=True) 
        return val

class RunningMeanStd(torch.nn.Module):
    def __init__(self, dim: int, clamp: float=0):
        super().__init__()
        self.epsilon = 1e-5
        self.clamp = clamp
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, x, unnorm=False):
        mean = self.mean.to(torch.float32)
        var = self.var.to(torch.float32)+self.epsilon # type: ignore
        if unnorm:
            if self.clamp:
                x = torch.clamp(x, min=-self.clamp, max=self.clamp)
            return mean + torch.sqrt(var) * x
        x = (x - mean) * torch.rsqrt(var)
        if self.clamp:
            return torch.clamp(x, min=-self.clamp, max=self.clamp)
        return x
    
    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, x.size(-1))
        var, mean = torch.var_mean(x, dim=0, unbiased=True)
        count = x.size(0)
        count_ = count + self.count
        delta = mean - self.mean
        m = self.var * self.count + var * count + delta**2 * self.count * count / count_ # type: ignore

        self.mean.copy_(self.mean+delta*count/count_) # type: ignore
        self.var.copy_(m / count_) # type: ignore
        self.count.copy_(count_) # type: ignore

    def reset_counter(self):
        self.count.fill_(1) # type: ignore