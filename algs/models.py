import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=.1, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.orthogonal_(layer.bias.data.view(1, -1), bias_const)
    return layer


def create_network(input_size, hidden_sizes, output_size, activation=nn.ReLU(), output_activation=None):
    layers = [layer_init(nn.Linear(input_size, hidden_sizes[0])), activation]
    for i in range(1, len(hidden_sizes)):
        layers.append(layer_init(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])))
        layers.append(activation)
    layers.append(layer_init(nn.Linear(hidden_sizes[-1], output_size)))

    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, config):
        super(Policy, self).__init__()
        self.std = config['std']
        self.pol_hidden_layer = config['pol_hidden']
        self.val_hidden_layer = config['val_hidden']
        self.policy_lr = config['policy_lr']
        self.value_lr = config['value_lr']
        self.policy_clip = config['policy_clip']
        self.distribution = torch.distributions.normal.Normal

        self.filter = RunningMeanStd(num_inputs)

        self.policy = create_network(num_inputs, self.pol_hidden_layer, num_outputs, activation=nn.ReLU(), output_activation=None)
        self.val_fn = create_network(num_inputs, self.val_hidden_layer, 1, activation=nn.ReLU(), output_activation=None)

        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = optim.AdamW(self.val_fn.parameters(), lr=self.value_lr)

    def forward(self, inputs, explore=True, update_filter=True):
        inputs = self.filter(inputs, update=update_filter)
        vf_pred = self.val_fn(inputs)
        mean = self.policy(inputs)

        action_dist = self.distribution(mean, self.std)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(-1)

        if explore == False:
            action = action_dist.loc

        return action, log_prob, vf_pred

    def value_loss(self, states, returns):
        states = self.filter(states, update=False)
        curr_values = self.val_fn(states)
        critic_loss = .5 * torch.nn.functional.mse_loss(curr_values.flatten(), returns)
        return critic_loss

    def policy_loss(self, states, actions, old_log_probs, returns):
        states = self.filter(states, update=False)
        with torch.no_grad():
            values = self.val_fn(states)

        advantages = (returns.reshape(-1, 1) - values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        mean_actions = self.policy(states)
        curr_action_dist = self.distribution(mean_actions, self.std)
        log_probs = curr_action_dist.log_prob(actions).sum(-1)

        ratio = torch.exp(log_probs - old_log_probs).unsqueeze(1)

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.policy_clip, 1 + self.policy_clip) * advantages
        actor_loss = -(torch.min(surr1, surr2)).mean()  # + .001 * (mean_actions ** 2).mean()

        return actor_loss

    def policy_step(self, loss):
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def value_step(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()


# http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
class RunningMeanStd:
    def __init__(self, shape, device='cuda'):
        self.device = device
        if shape is None:
            self.mean = np.zeros(1)
            self.std = np.zeros(1)
        else:
            self.n = torch.zeros(1, dtype=torch.float32, device=device)
            self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
            self.std = torch.ones(shape, dtype=torch.float32,  device=device)

        self.std_min = 0.03
        self.clip = 10.0

    def update(self, x):
        n1 = self.n
        n2 = x.shape[0]

        old_mean = self.mean
        old_std = self.std
        batch_mean = torch.mean(x, dim=0).to(self.device)
        batch_std = torch.std(x, dim=0).to(self.device)

        self.mean = n1 / (n1 + n2) * old_mean + n2 / (n1 + n2) * batch_mean
        S = n1 / (n1 + n2) * old_std ** 2 + n2 / (n1 + n2) * batch_std ** 2 + n1 * n2 / (n1 + n2) ** 2 * (old_mean - batch_mean) ** 2
        self.std = torch.sqrt(S)
        self.n += n2

    def __call__(self, x, update=True):
        if update:
            self.update(x)

        x = x.to(self.device) - self.mean.to(self.device)
        std = self.std.to(self.device)

        std = torch.clip(std, self.std_min, 1e5)

        x = x / (std + 1e-5)
        x = torch.clip(x, -self.clip, self.clip)
        return x

    def unfilter(self, x):
        x = torch.tensor(x).to(self.device)
        x = x * self.std
        x = x + self.mean
        return x

    def state_dict(self):
        state = {}
        state['n'] = self.n
        state['m'] = self.mean
        state['s'] = self.std
        return state

    def load_state_dict(self, state):
        self.n = state['n']
        self.mean = state['m']
        self.std = state['s']
