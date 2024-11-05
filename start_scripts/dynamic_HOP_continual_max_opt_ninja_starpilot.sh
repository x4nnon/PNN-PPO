#!/bin/bash

# FOR USE WITH HEX container is: tc2034/frapa2
# hare run -it --gpus device=8  -v "$(pwd)":/app tc2034/frapa2 bash

WANDB_PROJECT_NAME="ninja_starpilot_frapa_continued_learning_experiment2"

X_VALUES=(1 2 3 4)  # Specify the values for the seed here
# Loop over each combination of x and y


#     pnn (use number_hs=100 to distinguish)
python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed=1 \
--number_hs=19 --dynamic_weighting_of_learning_p=1 \
--track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
--total_timesteps=9000000 --env_id="procgen:procgen-ninja-v0" \
--int_env_id="procgen:procgen-starpilot-v0" --proc_sequential --min_similarity_score=0.98

python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed=2 \
--number_hs=19 --dynamic_weighting_of_learning_p=1 \
--track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
--total_timesteps=9000000 --env_id="procgen:procgen-ninja-v0" \
--int_env_id="procgen:procgen-starpilot-v0" --proc_sequential --min_similarity_score=0.98

python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed=3 \
--number_hs=19 --dynamic_weighting_of_learning_p=1 \
--track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
--total_timesteps=9000000 --env_id="procgen:procgen-ninja-v0" \
--int_env_id="procgen:procgen-starpilot-v0" --proc_sequential --min_similarity_score=0.98

python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed=4 \
--number_hs=19 --dynamic_weighting_of_learning_p=1 \
--track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
--total_timesteps=9000000 --env_id="procgen:procgen-ninja-v0" \
--int_env_id="procgen:procgen-starpilot-v0" --proc_sequential --min_similarity_score=0.98
    

