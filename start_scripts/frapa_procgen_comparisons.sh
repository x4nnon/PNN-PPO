WANDB_PROJECT_NAME="HOP_ProcGen_Comparisons_3_seq"

X_VALUES=(1 2 3)  # Specify the values for the seed here
# Loop over each combination of x and y


for x in "${X_VALUES[@]}"; do
    
    ###############################################################
    ENV_ID="procgen:procgen-fruitbot-v0"
    
    # benchmark
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID"\
    --number_hs=1 --min_similarity_score=1.1 --max_dict_in_compress=1 \
    --max_val_only=1 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --total_timesteps=5000000  --proc_sequential
    
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID" \
    --number_hs=20 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=7.5 --total_timesteps=5000000 --proc_sequential

    ###############################################################
    ENV_ID="procgen:procgen-starpilot-v0"
    
    # benchmark
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID"\
    --number_hs=1 --min_similarity_score=1.1 --max_dict_in_compress=1 \
    --max_val_only=1 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --total_timesteps=5000000 --proc_sequential
    
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID" \
    --number_hs=20 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=10 --total_timesteps=5000000--proc_sequential

    ###############################################################
    ENV_ID="procgen:procgen-ninja-v0"
    
    # benchmark
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID"\
    --number_hs=1 --min_similarity_score=1.1 --max_dict_in_compress=1 \
    --max_val_only=1 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --total_timesteps=5000000 --proc_sequential
    
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID" \
    --number_hs=20 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=9 --total_timesteps=5000000 --proc_sequential

    ###############################################################
    ENV_ID="procgen:procgen-climber-v0"
    
    # benchmark
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID"\
    --number_hs=1 --min_similarity_score=1.1 --max_dict_in_compress=1 \
    --max_val_only=1 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --total_timesteps=5000000 --proc_sequential
    
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID" \
    --number_hs=20 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=9 --total_timesteps=5000000 --proc_sequential

    ###############################################################
    ENV_ID="procgen:procgen-bigfish-v0"
    
    # benchmark
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID"\
    --number_hs=1 --min_similarity_score=1.1 --max_dict_in_compress=1 \
    --max_val_only=1 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --total_timesteps=5000000 --proc_sequential
    
    
    #     20
    python3 "$(pwd)/methods/frapa_ppo_continued_learning.py" --seed="$x" --env_id="$ENV_ID" --int_env_id="$ENV_ID" \
    --number_hs=20 --min_similarity_score=0.98 --max_dict_in_compress=0 \
    --max_val_only=0 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=1 --use_monochrome=0 \
    --reward_limit=6 --total_timesteps=5000000 --proc_sequential

    ###############################################################

    #
    
    

done






