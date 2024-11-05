#!/bin/bash


# FOR USE WITH HEX
# WE MESSED THE NAMING UP FOR THIS, IT IS ACTUALLY NINJA AND COINRUN

WANDB_PROJECT_NAME="ninja_coinrun_frapa_continued_learning_experiment3"

X_VALUES=(1 2 3 4)  # Specify the values for the seed here
# Loop over each combination of x and y


for x in "${X_VALUES[@]}"; do
    
    # benchmark
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" \
    --number_hs=1 --min_similarity_score=1.1 --max_dict_in_compress=1 --total_timesteps=9000000\
    --max_val_only=1 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=7.5 --env_id="procgen:procgen-ninja-v0" --int_env_id="procgen:procgen-coinrun-v0" --track --proc_sequential
    
    
    # 50
    # python3 "$(pwd)/methods/frapa_ppo.py" --seed="$x" --env_id="$ENV_ID" \
    # --number_hs=50 --min_similarity_score=0.9 --max_dict_in_compress=1 \
    # --max_val_only=1 --proc_num_levels=100 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=0 --use_monochrome=0 \
    # --reward_limit=7.5
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" \
    --number_hs=18 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=7.5 --total_timesteps=9000000 --env_id="procgen:procgen-ninja-v0" \
    --int_env_id="procgen:procgen-coinrun-v0" --track --proc_sequential


    #
    
    
done






