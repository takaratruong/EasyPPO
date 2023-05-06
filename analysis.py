import gym
import env  # GYM CANT FIND IT OTHERWISE
from algs.models import *
import os
import pprint


class Exp:
    def __init__(self, policy_path):
        self._config = None
        self.env = None
        self.policy = None
        self.model = None
        self.info = None

        self.init_policy(policy_path)

        self.save_folder = f"{os.path.dirname(policy_path)}/analysis"
        os.makedirs(self.save_folder, exist_ok=True)

    def init_policy(self, policy_path):
        exp_state = torch.load(policy_path)
        self.info = exp_state['info']
        print(self.info)
        self._config = exp_state['config']
        pprint.pprint(self._config, sort_dicts=False)

        env = gym.make(self._config['env']['env_id'], config=self._config, new_step_api=True)
        self.model = Policy(env.observation_space.shape[0], env.action_space.shape[0], self._config['policy'])
        self.model.load_state_dict(exp_state['policy'][0])
        self.model.filter.load_state_dict(exp_state['policy'][1])

        self.model.cuda()

    def visualize(self):
        env = gym.make(self._config['env']['env_id'], config=self._config, new_step_api=True, render_mode='human', autoreset=True)
        state = env.reset()

        while True:
            with torch.no_grad():
                action, _, _ = self.model(torch.as_tensor(state, dtype=torch.float32, device=device), explore=False, update_filter=False)
            next_state, reward, terminated, done, info = env.step(action.cpu())
            state = next_state.copy()

if __name__ == "__main__":

    policy_path ='results/models/exp_name/best_model.pt'

    exp_frame = Exp(policy_path)
    exp_frame.visualize()