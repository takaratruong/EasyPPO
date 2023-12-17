import gymnasium as gym
# import env
from algs.models import *
import os
import pprint


class Exp:
    def __init__(self, policy_path):
        self.policy = None
        self.model = None

        self.exp_state = torch.load(policy_path)
        self.config = self.exp_state['config']

        self.init_policy()

    def init_policy(self):
        print(self.exp_state['info'])
        pprint.pprint(self.config, sort_dicts=False)

        env = gym.make(self.config['env']['env_id'])
        self.model = Policy(env.observation_space.shape[0], env.action_space.shape[0], self.exp_state['config']['policy'])

        # Load model weights and observation normalization stats
        self.model.load_state_dict(self.exp_state['policy'][0])
        self.model.filter.load_state_dict(self.exp_state['policy'][1])

        self.model.cuda()

    def visualize(self):
        env = gym.make(self.config['env']['env_id'], render_mode='human')
        state, _ = env.reset()
        # import ipdb; ipdb.set_trace()
        while True:
            with torch.no_grad():
                action, _, _ = self.model(torch.as_tensor(state, dtype=torch.float32, device=device), explore=False, update_filter=False)
            next_state, reward, term, trunc, info = env.step(action.cpu())
            state = next_state.copy()

            if term or trunc:
                state, _ = env.reset()

if __name__ == "__main__":

    policy_path ='results/models/test/best_model.pt'

    exp_frame = Exp(policy_path)
    exp_frame.visualize()