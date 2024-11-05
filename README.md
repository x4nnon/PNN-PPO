# PNN-PPO with ProcGen

An implementation of Progressive Neural Networks (PNN) with Proximal Policy Optimization (PPO) updates, designed to work with the ProcGen suite of environments for generalization testing.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Citations](#citations)

## Introduction

This project implements Progressive Neural Networks (PNN) with PPO updates. PNNs are a neural network architecture designed to facilitate transfer learning by using lateral connections to transfer knowledge between tasks. PPO is a reinforcement learning algorithm that balances exploration and exploitation by using a clipped surrogate objective function.

The implementation is tailored to work with the ProcGen suite, a collection of procedurally generated environments that are used to test the generalization capabilities of reinforcement learning agents.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/x4nnon/pnn-ppo.git
cd pnn-ppo
pip install -r requirements.txt
```

Ensure you have the necessary environment setup for CUDA if you plan to use GPU acceleration.

## Usage

To run the training script, use the following command:

```bash
python methods/pnn_ppo_continued_learning_multi_new.py --arg1 value1 --arg2 value2
```


## Arguments

Below is a list of all the arguments you can pass to the script, along with their descriptions:

- `--exp_name`: Name of the experiment. Used for logging and tracking.
- `--seed`: Random seed for reproducibility.
- `--torch_deterministic`: If toggled, sets `torch.backends.cudnn.deterministic=False`.
- `--cuda`: Boolean flag to enable CUDA. Use `--cuda` to enable.
- `--track`: Boolean flag to enable tracking with Weights and Biases.
- `--wandb_project_name`: The wandb's project name.
- `--wandb_entity`: The entity (team) of wandb's project.
- `--total_timesteps`: Total number of timesteps for training.
- `--learning_rate`: Learning rate for the optimizer.
- `--num_envs`: Number of parallel environments.
- `--num_steps`: Number of steps per environment per update.
- `--anneal_lr`: Boolean flag to enable learning rate annealing.
- `--gamma`: Discount factor gamma.
- `--gae_lambda`: Lambda for the general advantage estimation.
- `--num_minibatches`: Number of mini-batches.
- `--update_epochs`: Number of epochs to update the policy.
- `--norm_adv`: Toggles advantages normalization.
- `--clip_coef`: Surrogate clipping coefficient.
- `--clip_vloss`: Toggles whether to use a clipped loss for the value function.
- `--ent_coef`: Coefficient of the entropy.
- `--vf_coef`: Coefficient of the value function.
- `--max_grad_norm`: Maximum norm for the gradient clipping.
- `--target_kl`: Target KL divergence threshold.
- `--anneal_ent`: Toggle entropy coefficient annealing.
- `--gen_traj`: A tag for generating a trajectory, not for testing.
- `--resnet_features`: 1 for true and 0 for false.
- `--debug`: Enables debug mode.
- `--proc_start`: Start level for ProcGen environments.
- `--proc_num_levels`: Number of levels for ProcGen environments.
- `--proc_sequential`: Whether to use sequential levels in ProcGen.
- `--proc_int_start`: Start level for new environment/continued learning.
- `--max_eval_ep_len`: Maximum evaluation episode length.
- `--start_ood_level`: Start level for out-of-distribution testing.
- `--env_ids`: Environment IDs to cycle between.
- `--cycles`: Number of times to cycle through the environment IDs.
- `--easy`: Difficulty setting for ProcGen environments.
- `--eval_interval`: Number of timesteps between evaluations.
- `--eval_specific_envs`: Number of specific environments to evaluate.
- `--eval_repeats`: Number of evaluation repeats.
- `--eval_batch_size`: Batch size for evaluations.
- `--use_monochrome`: Use monochrome assets in ProcGen.
- `--PNN`: Grouping flag for wandb.


### Example Command

```bash
python methods/pnn_ppo.py --exp_name "test_run" --seed 42 --cuda --track --env_ids "procgen:coinrun-v0,procgen:starpilot-v0" --total_timesteps 1000000 --learning_rate 0.0003 --num_envs 8 --num_steps 128 --anneal_lr --clip_coef 0.2 --ent_coef 0.01 --vf_coef 0.5 --max_grad_norm 0.5
```

## Citations

If you use this code in your research, please consider citing the following:

- **Progressive Neural Networks**: Rusu, A. A., et al. "Progressive neural networks." arXiv preprint arXiv:1606.04671 (2016).
- **Proximal Policy Optimization**: Schulman, J., et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
- **ProcGen**: Cobbe, K., et al. "Leveraging procedural generation to benchmark reinforcement learning." arXiv preprint arXiv:1912.01588 (2019).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
