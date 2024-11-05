#!/bin/bash


# FOR USE WITH HEX container is: tc2034/frapa2
# hare run -it --gpus device=8  -v "$(pwd)":/app tc2034/frapa2 bash

WANDB_PROJECT_NAME="ninja_coinrun_frapa_continued_learning_experiment3"

X_VALUES=(1 2 3 4)  # Specify the values for the seed here
# Loop over each combination of x and y

    
#     pnn (use number_hs=100 to distinguish)
python3 "$(pwd)/methods/pnn_ppo_continued_learning1.py" --seed=1 \
--number_hs=101 \
--track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
--total_timesteps=9000000 --env_id="procgen:procgen-ninja-v0" \
--int_env_id="procgen:procgen-coinrun-v0" --proc_sequential

python3 "$(pwd)/methods/pnn_ppo_continued_learning1.py" --seed=2 \
--number_hs=101 \
--track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
--total_timesteps=9000000 --env_id="procgen:procgen-ninja-v0" \
--int_env_id="procgen:procgen-coinrun-v0" --proc_sequential

python3 "$(pwd)/methods/pnn_ppo_continued_learning1.py" --seed=3 \
--number_hs=101 \
--track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
--total_timesteps=9000000 --env_id="procgen:procgen-ninja-v0" \
--int_env_id="procgen:procgen-coinrun-v0" --proc_sequential

python3 "$(pwd)/methods/pnn_ppo_continued_learning1.py" --seed=4 \
--number_hs=101 \
--track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
--total_timesteps=9000000 --env_id="procgen:procgen-ninja-v0" \
--int_env_id="procgen:procgen-coinrun-v0" --proc_sequential
    






