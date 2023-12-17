config = {
    'wandb': 'online', # disabled

    'num_envs': 100,
    'num_steps': 50,
    'max_iter': 12000,
    'seed':10, 

    'logging': {
        'exp_name': 'test',
        'project': 'easy-ppo',
        'save_path': '/results/models/',
        'vid_rec_frq': 50,
    },

    'env': {
        'env_id': 'Humanoid-v4', # works with any gym environment. If using mujoco envs like (Humanoid-v4), downgrade to mujoco 2.3.*
        },

    'policy': {
        'num_epochs': 10,
        'batch_size': 1250,
        'pol_hidden': [1024, 1024],
        'val_hidden': [128, 128],
        'std': 0.08208,  
        'gamma': 0.99, 
        'lambda': 0.95, 
        'policy_lr': 1e-5, 
        'value_lr': 1e-4, 
        'policy_clip': 0.2,
        },
}