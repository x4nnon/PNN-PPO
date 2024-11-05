WANDB_PROJECT_NAME="HOP_ProcGen_Comparisons"

X_VALUES=(1 2 3)  # Specify the values for the seed here
# Loop over each combination of x and y


for x in "${X_VALUES[@]}"; do

    ###############################################################
    ENV_ID="procgen-ninja"
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID" \
    --number_hs=20 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=8.5 --total_timesteps=5000000

    ###############################################################

    ###############################################################
    ENV_ID="procgen-starpilot"

    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID" \
    --number_hs=20 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=15 --total_timesteps=5000000
    
    ###############################################################
    ENV_ID="procgen-climber"
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID" \
    --number_hs=20 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=9 --total_timesteps=5000000
    
    
    ###############################################################
    ENV_ID="procgen-coinrun"
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID" \
    --number_hs=20 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=7.5 --total_timesteps=5000000

    

    ###############################################################
    ENV_ID="procgen-bossfight"
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID" \
    --number_hs=20 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=2 --total_timesteps=5000000

    

    #

done






