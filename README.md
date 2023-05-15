# EasyPPO
EasyPPO is a straightforward and user-friendly implementation of Proximal Policy Optimization (PPO), offering integration with a custom Mujoco environment and Weights & Biases (WandB) logging. This repository is designed for simplicity and serves as an ideal starting point for researchers new to PPO or those seeking a comprehensible implementation.

## Overview
Originating from The Movement Lab at Stanford, EasyPPO was initially created for character animation. However, it has been successfully repurposed for robotic arm control envs as well. The only prerequisite is a compatible gym environment. Its primary features include:

* A clean, minimalist PPO implementation suitable for both beginners and those seeking a simplified codebase.
* An example of a custom Mujoco environment (training a humanoid to walk using DeepMimic). 
* Integrated WandB logging for effortless performance tracking and visualization during training.

## Getting Started
To get started with this implementation, follow these steps:
1. Clone the repository
2. Install the required dependencies
3. Run the example script
4. Customize the codebase for your specific needs

## Known Issues
* The code was developed with gym 0.25.2 and needs wandb 0.14.0 to work. 

## Future Work 
1. Update environment and code to support the latest version of gymnasium.
2. Implement the obs-normalization wrapper in place of the custom solution.
