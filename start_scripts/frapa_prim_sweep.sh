ENV_ID="procgen-coinrun"
WANDB_PROJECT_NAME="CoinRun_frapa_math"

X_VALUES=(1 2 3 4 5)  # Specify the values for the seed here
# Loop over each combination of x and y

for x in "${X_VALUES[@]}"; do
    
    # benchmark
    python3 "$(pwd)/methods/frapa_ppo.py" --seed="$x" --env_id="$ENV_ID" \
    --number_hs=1 --min_similarity_score=1.1 --max_dict_in_compress=1 \
    --max_val_only=1 --proc_num_levels=20 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=0 --use_monochrome=0 \
    --reward_limit=8.5

done

for x in "${X_VALUES[@]}"; do

    ########## And all the same but at more procgen levels
    
    # benchmark
    python3 "$(pwd)/methods/frapa_ppo.py" --seed="$x" --env_id="$ENV_ID" \
    --number_hs=1 --min_similarity_score=1.1 --max_dict_in_compress=1 \
    --max_val_only=1 --proc_num_levels=100 --track --wandb_project_name="$WANDB_PROJECT_NAME" --easy=0 --use_monochrome=0 \
    --reward_limit=8.5
    
    
done
