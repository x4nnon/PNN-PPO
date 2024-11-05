
#!/bin/bash


# FOR USE WITH HEX

WANDB_PROJECT_NAME="HOP_multi_ninja_starpilot_3cycle"

X_VALUES=(1 2 3 4)  # Specify the values for the seed here
# Loop over each combination of x and y


for x in "${X_VALUES[@]}"; do
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning_multi.py" --seed="$x" \
    --number_hs=33 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=7.5 --all_optimizers --total_timesteps 12000000 --env_id="procgen:procgen-fruitbot-v0" \
     --int_env_id="procgen:procgen-coinrun-v0" --track \
     --env_ids "procgen:procgen-ninja-v0,procgen:procgen-starpilot-v0" --cycles 3

    #
    
    
done










