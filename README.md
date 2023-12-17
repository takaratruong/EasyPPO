# EasyPPO
EasyPPO is a straightforward and user-friendly implementation of Proximal Policy Optimization (PPO), offering integration with Weights & Biases (WandB) logging. This repository is designed for simplicity and serves as an ideal starting point for researchers new to PPO or those seeking a comprehensible implementation.

## Overview
Originating from The Movement Lab at Stanford, EasyPPO was initially created for methods in character animation using mujoco. However, it has been successfully used for robotic arm control in other physics engines (pybullet). The only prerequisite is a compatible gym environment. Its primary features include:

* A clean, minimalist PPO implementation suitable for both beginners and those seeking a simplified codebase.
* Integrated WandB logging for effortless performance tracking and visualization during training.

## Getting Started
To get started with this implementation, follow these steps:
1. Clone the repository
2. Install the required python version + dependencies
   1. python=3.11.*
   2. gymnasium=0.29.1
   3. mujoco=2.3.7
   4. pytorch  
   5. wandb
   
3. Run the example script to train the model (run.py) 
4. Load and run the trained model (analysis.py)

