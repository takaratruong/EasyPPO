config = {
    'exp_name': 'juggle_test',
    'wandb': 'online',

    'seed':0,
    'num_envs': 100,  
    'num_steps': 100,  
    'batch_size': 500, 

    'max_epochs': 5e10,
    'opt_epochs': 10,    
    
    'state_dim': 34,
    'goal_dim': 1, 
    'rew_val_dim': 2,
    'normalize_values': True, 

    'term_reward': None, # Overrides reward at termination, make sure that this value is less than the sum of cumulative rewards otherwise a sufficient strategy will be to unalive itself
    'zero_term_next_val': False, # If the episode terminates then the next_value should be zero. For challenging taks where there is termination before a good enough reward, zero'ing can make it even harder for rewards to propogate. 
    'explore_noise': .1, # None for learned explore noise 
    'gamma': 0.99,
    'lambda': 0.95, 
    'policy_lr': 1e-5, 
    'value_lr': 1e-4,
    
    'env': {
        'xml_file': 'juggle/assets/model.xml',
        'train': True,
        'obs_horizon':4, 
        'obs_size': 34*4+1, #277 # this is for gymnasium because it does some internal checking
        'z_goal': 2.0, 
        'z_0': 1.0,
        'frame_skip': 15,         
        'env_id': 'juggle',   
        'max_ep_time': 20,
        'diffusion':False,      
    },
    
    'logging': {
        'project': 'Ping-Pong',
        'save_path': '/move/u/takaraet/ping_pong/results/models/',
        'vid_rec_frq': 50,
    },    
}
