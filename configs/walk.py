config = {
    'exp_name': 'norm adv',
    'train': True,
    'wandb': 'online',

    'num_envs': 100,
    'num_steps': 50,
    'max_iter': 12000,

    'logging': {
        'project': 'motion_vae',
        'save_path': '/home/motion_vae/results/models/',
        'vid_rec_frq': 50,
    },
    'env': {
        'ref_path': 'env/motion/walk',
        'frame_skip': 2,
        'max_ep_time': 6,
        'env_id': 'Skeleton',
        'xml_file': 'assets/skeleton_model.xml',
    },
    'policy': {
        'obs_size': 311,
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