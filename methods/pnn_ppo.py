#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:46:53 2024

@author: x4nno
"""

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import time
from dataclasses import dataclass
from datetime import datetime
import sys

sys.path.append("/home/x4nno/Documents/PhD/MetaGridEnv/MetaGridEnv")
sys.path.append("/home/x4nno/Documents/PhD/FraPA")

# sys.path.append("/app/MetaGridEnv/MetaGridEnv")
sys.path.append("/app/FraPA")
sys.path.append("/home/x4nno_desktop/Documents/MetaGridEnv/MetaGridEnv")
sys.path.append("/home/x4nno_desktop/Documents/FraPA")

sys.path.append("/app")

from gym import Wrapper
import gym as gym_old # for procgen 

import ast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import copy
from gym.envs.registration import register 
from procgen import ProcgenEnv
from PIL import Image
from functools import reduce

from matplotlib import pyplot as plt

register( id="MetaGridEnv/metagrid-v0",
          entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")

from gymnasium.vector import SyncVectorEnv

from agents.pnn_agent import PNNAgent

from utils.compatibility import EnvCompatibility

from pympler import asizeof

# needed for atari if you decide to use
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)



def vis_env_master(envs):
    plt.imshow(envs.envs[0].env_master.domain)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True #!!! change to True for running realtime
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "PNN_testing_new"
    """the wandb's project name"""
    wandb_entity: str = "tpcannon"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = ""
    """the id of the environment: computed from the list of env_ids"""
    total_timesteps: int = 4000000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = True #always true
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False # this was True when gathering good results.
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.05
    """the target KL divergence threshold"""
    report_epoch: int = num_steps*num_envs*10
    """When to run a seperate epoch run to be reported. Make sure this is a multple of num_envs."""
    anneal_ent: bool = False
    """Toggle entropy coeff annealing"""


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    

    gen_traj: bool = False # MUST CHANGE THIS to FALSE BEFORE RUNNING ELSE WONT GET ANY RESULTS FROM EVALS!
    """ A tag which should be used if we are generating a trajectory, not for testing. """
    resnet_features: int = 1 # 1 for true and 0 for false
    
    debug: bool = False # makesure to change to False before running
    
    #env_specific
    proc_start: int = 1 # 0 for train. > 50 for test
    proc_num_levels: int = 30 # 50 for train. > for test
    proc_sequential: bool = True
    proc_int_start: int = 1 # the level to start on when we swap for a new environment / continued learning 
    
    max_eval_ep_len: int = 1000
    specific_proc_list_input: str = "None"
    specific_proc_list = ast.literal_eval(specific_proc_list_input)
    
    
    # frapa specific
    number_hs: int = 100
    """The total number of checkpoints or hierarchy levels used with HOP but can be used to 
    distinguish PNN-PPO runs from HOP"""

    start_ood_level: int = 42000
    env_ids: str = "procgen-plunder,procgen-starpilot"
    """The environment ids to cycle between"""
    cycles: int = 2
    """The number of times to cycle through the env_ids"""
    #procgen specific
    easy: int = 1 # 1 = True and 0 = False and 2 = Exploration
    
    eval_interval: int = 100000
    """The number of timesteps between evaluations"""
    #eval
    eval_specific_envs: int = 0
    eval_repeats: int = 1
    eval_batch_size: int = 30 # must divide the proc_num_levels
    
    use_monochrome: int = 0 # 0 for false, 1 for true
    
    PNN: int = 1 # just to group in wandb
    max_columns: int = 0
    
    if debug:
        # num_envs=1
        report_epoch = num_steps*num_envs
    
    if max_eval_ep_len == 0:
        max_eval_ep_len = max_ep_length
    
    
def normalize(tensor):
    norm = torch.linalg.norm(tensor)
    return tensor / norm if norm != 0 else tensor


def plot_all_procgen_obs(next_obs, envs):
    try:
        im_d = next_obs.to("cpu")
    except:
        im_d = next_obs
    im_d_orig = np.array(im_d)
    for i in range(len(envs.envs)):
        im_d = im_d_orig[i]
        im_d = im_d/255.0
        plt.imshow(im_d)
        plt.axis("off")
        plt.show()
        
        
def plot_specific_procgen_obs(next_obs, i):
    try:
        im_d = next_obs.to("cpu")
    except:
        im_d = next_obs
    im_d_orig = np.array(im_d)
    if len(im_d_orig.shape) > 3:
        im_d = im_d_orig[i]
    else:
        im_d = im_d_orig
    im_d = im_d/255.0
    plt.imshow(im_d)
    plt.axis("off")
    plt.show()
        

def make_env(env_id, idx, capture_video, run_name, args, sl=1, nl=10, enforce_mes=False, easy=True, seed=0):
    def thunk():
        
        if args.specific_proc_list:
            sl_in = random.choice(args.specific_proc_list)
            # print(sl_in)
            nl_in=1
        else: # these need to change later -- why are errors happening with sl and nl
            sl_in = sl
            nl_in= nl
        
        if "procgen" in args.env_id: # need to create a specific method for making these vecenvs
            # The max ep length is handled in the fracos wrapper - the envcompatibility will give warnings about 
            # early reset being ignored, but it does get truncated by the fracos_wrapper. So you can ignore these warnings.
            # print(sl_in)
            if easy:
                # print(sl_in)
                if args.use_monochrome:
                    env = gym_old.make(args.env_id, num_levels=nl_in, start_level=sl_in, distribution_mode="easy", use_backgrounds=False, restrict_themes=True, use_monochrome_assets=True, use_sequential_levels=args.proc_sequential, rand_seed=int(seed)) # change this will only do one env
                else:
                    env = gym_old.make(args.env_id, num_levels=nl_in, start_level=sl_in, distribution_mode="easy", use_backgrounds=False, restrict_themes=True, use_monochrome_assets=False, use_sequential_levels=args.proc_sequential, rand_seed=int(seed)) # change this will only do one env
            else:
                # print(sl_in)
                if args.use_monochrome:
                    env = gym_old.make(args.env_id, num_levels=nl_in, start_level=sl_in, distribution_mode="hard", use_backgrounds=False, restrict_themes=True, use_monochrome_assets=True, use_sequential_levels=args.proc_sequential, rand_seed=int(seed)) # change this will only do one env
                else:
                    env = gym_old.make(args.env_id, num_levels=nl_in, start_level=sl_in, distribution_mode="hard", use_backgrounds=False, restrict_themes=True, use_monochrome_assets=False, use_sequential_levels=args.proc_sequential, rand_seed=int(seed)) # change this will only do one env
            
            env.observation_space = gym.spaces.Box(0,255,(64,64,3), "int")
            env.action_space = gym.spaces.Discrete(env.action_space.n)
            
            # env.action_space
            #envs = gym_old.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
            
            env = EnvCompatibility(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym_old.wrappers.NormalizeReward(env, gamma=args.gamma)
            env = gym_old.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            
            ## Needs a wrapper to turn to from gym.spaces to gymnasium.spaces for both obs and action?
            
        elif "atari" in env_id:
            env_in_id = env_id.split(":")[-1] # because we only want the name but to distinguish we need the atari:breakout etc
            env = gym.make(env_in_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env = gym.wrappers.RecordEpisodeStatistics(env)
            # if capture_video:
            #     if idx == 0:
            #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
            
        
        else:
            env = gym.make(env_id, max_episode_steps=args.max_ep_length)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        

        return env

    return thunk

 
            
        
def conduct_evals(agent, writer, global_step_truth, run_name, device, current_column, int_level=False):
    first_ep_rewards = np.full(args.eval_specific_envs, None, dtype=object)
    first_ep_success = np.full(args.eval_specific_envs, None, dtype=object)
    cum_first_ep_rewards = np.zeros(args.eval_specific_envs)
    rep_summer = 0
    success_summer = 0
    success_list = []
    cbs = False
    for rep in range(args.eval_repeats):
        with torch.no_grad():
            ## Get the eval eps.
            # we need to randomise the sl because at the moment we only test on the same level everytime!!!
            if not int_level:
                sl_counter = args.proc_start
                start_level = args.proc_start
            else:
                sl_counter = args.proc_int_start
                start_level = args.proc_int_start
                
            for i in range(0, args.eval_specific_envs, args.eval_batch_size):
                print(sl_counter, "starting")
                    
                sls = [sl_counter + i for i in range(args.eval_batch_size)]
                
                if args.proc_sequential:
                    cbs = True
                    args.proc_sequential = False
                test_envs = SyncVectorEnv(
    				[make_env(args.env_id, sl, args.capture_video, run_name, args, sl=sl, nl=1, enforce_mes=True, easy=args.easy) for sl in sls],)
                if cbs:
                    args.proc_sequential = True
                # test_envs.train = True
                # test_seed = random.randint(1001,2000) #makes sure that the random int for train is below 1000 #handled seperatly
                test_next_obs, _ = test_envs.reset()
                test_next_obs = torch.Tensor(test_next_obs).to(device)
                all_test_rewards = 0
                for ts in range(args.max_eval_ep_len+1):
                    # print(ts)
                    test_action, test_logprob, _, test_value = agent.get_action_and_value(test_next_obs, current_column)
                    test_next_obs, test_reward, test_terminations, test_truncations, test_infos = test_envs.step(test_action.cpu().numpy())
                    test_next_obs = torch.Tensor(test_next_obs).to(device)
                    
                    for ve in range(len(test_reward)):
                        cum_first_ep_rewards[ve+(sl_counter-start_level)] += test_reward[ve]
                        if test_terminations[ve] or test_truncations[ve]:
                            if first_ep_rewards[ve+(sl_counter-start_level)] == None:
                                if "procgen" in args.env_id:
                                    first_ep_success[ve+(sl_counter-start_level)] = test_infos["final_info"][ve]["episode"]["r"].item()
                                    first_ep_rewards[ve+(sl_counter-start_level)] = test_infos["final_info"][ve]["episode"]["r"].item()
                                else:
                                    first_ep_rewards[ve+(sl_counter-start_level)] = cum_first_ep_rewards[ve+(sl_counter-start_level)]
                                
                    if all(val != None for val in first_ep_rewards):
                        break # because we don't want to keep stepping if all complete.
                
                for ve in range(len(test_reward)):
                    if first_ep_rewards[ve+(sl_counter-start_level)] == None:
                        first_ep_rewards[ve+(sl_counter-start_level)] = -(args.max_eval_ep_len+1)*0.01
                        first_ep_success[ve+(sl_counter-start_level)] = 0
                sl_counter += args.eval_batch_size
            rep_summer += sum(first_ep_rewards)
            success_summer += sum(first_ep_success)

                    
        
    writer.add_scalar("charts/avg_IID_eval_ep_rewards", rep_summer/(len(first_ep_rewards)*args.eval_repeats), global_step_truth)
    del test_envs
    
    ## OOD
    first_ep_rewards = np.full(args.eval_specific_envs, None, dtype=object)
    first_ep_success = np.full(args.eval_specific_envs, None, dtype=object)
    cum_first_ep_rewards = np.zeros(args.eval_specific_envs)
    rep_summer = 0
    success_summer = 0
    for rep in range(args.eval_repeats):
        with torch.no_grad():
            ## Get the eval eps.
            # we need to randomise the sl because at the moment we only test on the same level everytime!!!
            sl_counter = args.start_ood_level
            for i in range(0, args.eval_specific_envs, args.eval_batch_size):
                print(sl_counter, "starting")
                    
                sls = [sl_counter + i for i in range(args.eval_batch_size)]
    
                if args.proc_sequential:
                    args.proc_sequential = False
                    cbs = True
                test_envs = SyncVectorEnv(
    				[make_env(args.env_id, sl, args.capture_video, run_name, args, sl=sl, nl=1, enforce_mes=True, easy=args.easy) for sl in sls],)
                if cbs:
                    args.proc_sequential = True
                
                # test_envs.train = True
                # test_seed = random.randint(1001,2000) #makes sure that the random int for train is below 1000 #handled seperatly
                test_next_obs, _ = test_envs.reset()
                test_next_obs = torch.Tensor(test_next_obs).to(device)
                all_test_rewards = 0
                for ts in range(args.max_eval_ep_len+1):
                    # print(ts)
                    test_action, test_logprob, _, test_value = agent.get_action_and_value(test_next_obs, current_column)
                    test_next_obs, test_reward, test_terminations, test_truncations, test_infos = test_envs.step(test_action.cpu().numpy())
                    test_next_obs = torch.Tensor(test_next_obs).to(device)
                    
                    for ve in range(len(test_reward)):
                        cum_first_ep_rewards[ve+(sl_counter-args.start_ood_level)] += test_reward[ve]
                        if test_terminations[ve] or test_truncations[ve]:
                            if first_ep_rewards[ve+(sl_counter-args.start_ood_level)] == None:
                                if "procgen" in args.env_id:
                                    first_ep_success[ve+(sl_counter-args.start_ood_level)] = test_infos["final_info"][ve]["episode"]["r"].item()
                                    first_ep_rewards[ve+(sl_counter-args.start_ood_level)] = test_infos["final_info"][ve]["episode"]["r"].item()
                                else:
                                    first_ep_rewards[ve+(sl_counter-args.args.start_ood_level)] = cum_first_ep_rewards[ve+(sl_counter-args.start_ood_level)]
                                
                    if all(val != None for val in first_ep_rewards):
                        break # because we don't want to keep stepping if all complete.
                for ve in range(len(test_reward)):
                    if first_ep_rewards[ve+(sl_counter-args.start_ood_level)] == None:
                        first_ep_rewards[ve+(sl_counter-args.start_ood_level)] = -(args.max_eval_ep_len+1)*0.01
                        first_ep_success[ve+(sl_counter-args.start_ood_level)] = 0

                sl_counter += args.eval_batch_size
            
            rep_summer += sum(first_ep_rewards)
            success_summer += sum(first_ep_success)
                    
        
    writer.add_scalar("charts/avg_OOD_eval_ep_rewards", rep_summer/(len(first_ep_rewards)*args.eval_repeats), global_step_truth)
    del test_envs


def show_procgen(next_obs, env_num, global_decisions, action):
    
    try:
        im_d = next_obs.to("cpu")
    except:
        im_d = next_obs
    im_d = np.array(im_d)
    im_d = im_d[env_num]
    im_d = im_d/255.0
    plt.imshow(im_d)
    plt.axis("off")
    plt.title(f"step {global_decisions} action is {action[env_num]}")
    plt.show()
    
    
def show_good_obs(good_obs, good_actions):
    for i in range(len(good_obs)):
        obs = good_obs[i].cpu()
        obs = np.array(obs[0])/255.0
        plt.imshow(obs)
        plt.axis("off")
        plt.title(f"action is {good_actions[i]}")
        plt.show()
        
        
def freeze_non_active_columns(agent, current_column):
    for cidx in range(args.max_columns):
        if cidx != current_column: 
            for param in agent.actors[cidx].parameters():
                param.requires_grad = False
            for param in agent.critics[cidx].parameters():
                param.requires_grad = False
        if cidx == current_column:
            for param in agent.actors[cidx].parameters():
                param.requires_grad = True
            for param in agent.critics[cidx].parameters():
                param.requires_grad = True
                
    return agent


def pnn_ppo(args):    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    env_id_list_ind = args.env_ids.split(",")
    args.env_id = env_id_list_ind[0]
    args.max_columns = len(env_id_list_ind)
    env_id_list = env_id_list_ind * args.cycles
    column_names_to_id = {env_id_list_ind[i]: i for i in range(len(env_id_list_ind))}

    change_inc = args.num_iterations//len(env_id_list)
    print(change_inc)
    
    run_name = f"PNN_{env_id_list_ind}__{args.exp_name}__{args.seed}__{datetime.now()}"
    if args.track and not args.debug:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    proc_rng = np.random.default_rng(seed=42)
    proc_level_order_seeds = proc_rng.integers(low=0, high=10000, size=args.num_envs)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    
    seeds = proc_level_order_seeds
    
    envs = SyncVectorEnv(
        [make_env(env_id_list_ind[0], seed, args.capture_video, run_name, args, sl=args.proc_start, nl=args.proc_num_levels, easy=args.easy, seed=seed) for seed in seeds],
    )
    int_level = False
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    ####### for PNN differnces you want to look mostly here i'll try tag PNN in places where it changes ####
    agent = PNNAgent(input_channels=3, num_actions=15, hidden_dim=256, num_columns=2)
    # above -- only two columns despite 3 tasks. 
    agent_params = agent.parameters()
    
    optimizer = optim.Adam(agent_params, lr=args.learning_rate, eps=1e-5)

    agent.optimizers = [optimizer]
    agent.params = [agent_params]
    
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    resnet_results = torch.zeros((args.num_steps, args.num_envs, 512)).to(device)
    obs_np_flat = np.zeros((args.num_steps, args.num_envs, reduce(lambda x, y: x * y, envs.single_observation_space.shape)), dtype=np.int64) # dsc speed up
    
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    global_decisions = 0 
    global_step_truth = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs_np_flat = next_obs.reshape(args.num_envs, -1) # dsc speed up
    next_obs = torch.Tensor(next_obs).to(device)
    # change back
    next_done = torch.zeros(args.num_envs).to(device)

    epoch_res_count = 0
    cum_training_rewards = np.zeros(args.num_envs)
    # epoch_step_tracker = 0
    
    
    #PNN logic here --- freeze all columns that aren't the current_column
    current_column = 0
    agent = freeze_non_active_columns(agent, current_column)
    
    
    # initial results before learning:
    if (not args.debug) and (not args.gen_traj):
        # evaluation function
        conduct_evals(agent, writer, global_step_truth, run_name, device, current_column, int_level=int_level)
        
        epoch_res_count += 1
    
    # start the learning
    
    steper = args.num_iterations // args.number_hs  # Use integer division
    compression_intervals = [steper * (i + 1) for i in range(args.number_hs)] # !!! check this !!!
    level = 0
    for iteration in range(1, args.num_iterations + 1):
        
        if iteration % change_inc == 0:
            # PNN logic here -- change to the next column, freeze params of the first
            level += 1
            print(f"Old env id was {args.env_id}")
            env_id_temp = env_id_list[level]
            current_column = column_names_to_id[env_id_temp]

            agent = freeze_non_active_columns(agent, current_column)
            
            args.env_id = env_id_temp

            print(f"New env id is {args.env_id}")   

            if args.proc_sequential:
                envs = SyncVectorEnv(
                    [make_env(args.env_id, seed, args.capture_video, run_name, args, sl=args.proc_int_start, nl=0, easy=args.easy, seed=seed) for seed in seeds],
                )
            else:
                envs = SyncVectorEnv(
                    [make_env(args.env_id, seed, args.capture_video, run_name, args, sl=args.proc_int_start, nl=args.proc_num_levels, easy=args.easy, seed=seed) for seed in seeds],
                )
            next_obs, _ = envs.reset(seed=args.seed)
            next_obs_np_flat = next_obs.reshape(args.num_envs, -1) # dsc speed up
            next_obs = torch.Tensor(next_obs).to(device)
            # change back
            next_done = torch.zeros(args.num_envs).to(device)
            
            int_level = True

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (global_decisions/ args.total_timesteps)
            lrnow = frac * args.learning_rate
            first = True
            for optimmer in agent.optimizers:
                if first:
                    optimmer.param_groups[0]["lr"] = lrnow
                    first = False
                else:
                    optimmer.param_groups[0]["lr"] = lrnow
        if args.anneal_ent:
            frac = 1.0 - (global_decisions / args.total_timesteps)
            ent_coef_now = frac * args.ent_coef
        else:
            ent_coef_now = args.ent_coef
            
        for step in range(0, args.num_steps):
            
            # if args.debug:
            #     print(asizeof.asized(agent, detail=1).format())
            #     print(asizeof.asized(envs, detail=1).format())
            #     print("---------")
            
            obs[step] = next_obs
            obs_np_flat[step] = next_obs_np_flat # dsc speed up

            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # b4 = time.time()
                action, logprob, _, value= agent.get_action_and_value(next_obs, current_column)
                # after = time.time()
                # print(after-b4)
                values[step] = value.flatten()
            actions[step] = action

            logprobs[step] = logprob                
                
            if (args.debug) & ("procgen" in args.env_id):
                show_procgen(next_obs, 0, global_decisions, action)
                pass
            
            if args.num_envs == 1:
                action = action.unsqueeze(0)
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                       
            next_done = np.logical_or(terminations, truncations)
            
            cum_training_rewards += reward            
            
            global_decisions += args.num_envs
            
            # next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs_np_flat = next_obs.reshape(args.num_envs, -1) # dsc speed up
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            # next_done = torch.Tensor(next_done).to(device)
            # change back
            
            if "final_info" in infos:
                ref=0
                for info in infos["final_info"]:
                    if info and ("episode" in info):
                        print(f"global_step={global_decisions}, ep_r={info['episode']['r']}, ep_l={info['episode']['l']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_decisions)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_decisions)
                        # plot_specific_procgen_obs(next_obs, envs, ref)
                    ref += 1
            
            # where we get our epoch results from and we take the first ep as evals
            
            if (global_decisions == args.report_epoch*epoch_res_count) and (not args.gen_traj):
                # without running complete new trials
                all_test_rewards = sum(cum_training_rewards)
                # average_all_test_rewards = all_test_rewards/epoch_step_tracker
                writer.add_scalar("charts/epoch_returns", all_test_rewards, global_decisions)
                # writer.add_scalar("charts/average_return_per_epoch_step", average_all_test_rewards, global_decisions)
                
                cum_training_rewards = np.zeros(args.num_envs)

                # EVALS                
                conduct_evals(agent, writer, global_step_truth, run_name, device, current_column, int_level)
                epoch_res_count += 1
            
        b4 = time.time() # remove
        # bootstrap value if not done
        with torch.no_grad():
            if not torch.is_tensor(next_obs):
                next_obs = torch.tensor(next_obs).to(device)
            # 
            
            #
            next_value = agent.get_value(next_obs, current_column).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                # reward adjust (should be a wrapper but results gathered using this)
                adjusted_rewards = rewards[t]
                
                delta = adjusted_rewards + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_resnet_results = resnet_results.reshape((-1,512))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], current_column, action=b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef
                
                for optimmer in agent.optimizers:
                    optimmer.zero_grad()

                loss.backward()
                idx = 0
                for optimmer in agent.optimizers:
                    nn.utils.clip_grad_norm_(agent.params[idx], args.max_grad_norm)
                    optimmer.step()
                    idx += 1

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_decisions)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_decisions)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_decisions)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_decisions)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_decisions)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_decisions)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_decisions)
        writer.add_scalar("losses/explained_variance", explained_var, global_decisions)
        print("SPS:", int(global_decisions/ (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_decisions / (time.time() - start_time)), global_decisions)
        writer.add_scalar("charts/decisions", global_decisions, global_decisions)

        after = time.time()
        print("Time taken: ", after-b4)
        
        # This is now here, so is only removed after doing the update.
        if ("procgen" in args.env_id) or ("atari" in args.env_id): # because of conv hiddens
            agent.discrete_search_cache={} # the values are too big and rinsing memory too fast
        

        if (global_decisions >= args.total_timesteps):
            if not args.gen_traj:
                # without running complete new trials
                all_test_rewards = sum(cum_training_rewards)
                # average_all_test_rewards = all_test_rewards/epoch_step_tracker
                writer.add_scalar("charts/epoch_returns", all_test_rewards, global_decisions)
                # writer.add_scalar("charts/average_return_per_epoch_step", average_all_test_rewards, global_decisions)
                
                cum_training_rewards = np.zeros(args.num_envs)
                # epoch_step_tracker = 0
                
                conduct_evals(agent, writer, global_step_truth, run_name, device, current_column, int_level)
            epoch_res_count += 1
            break
        
    envs.close()
    
    #absolute final evals
    conduct_evals(agent, writer, global_step_truth, run_name, device, current_column, int_level)
    
    writer.close()
    

    return agent

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.eval_specific_envs == 0:
        args.eval_specific_envs = args.proc_num_levels
        
    args.specific_proc_list = ast.literal_eval(args.specific_proc_list_input)
    # args.debug=True
    
    
    pnn_ppo(args)