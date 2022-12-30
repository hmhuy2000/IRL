python train_PPO_WC.py --env_name=Safexp-PointGoal1-v0 --buffer_size=30000 --eval_interval=60000 \
--gamma=0.99 --reward_factor=1 --seed=1 --lr_actor=0.0003 --lr_critic=0.0001 --lr_cost_critic=0.001 \
--hidden_units_actor=256 --cost_limit=25 --lr_penalty=0.0001 \
--hidden_units_critic=256 --lambd=0.97 --epoch_ppo=80 \
--num_training_step=10000000 --num_envs=30 \
--wandb_logs=True \

# python render_PPO.py --env_name=Safexp-PointGoal1-v0 --buffer_size=100000 --seed=0 \
#  --hidden_units_actor=128 --hidden_units_critic=128 --max_episode_length=1000 \
#  --load_dir='./weights/PPO_lag/Safexp-PointGoal1-v0-23.99' \
#  --hidden_units_actor=256 --cost_limit=25 --lr_penalty=0.0001 \
#  --hidden_units_critic=256 --lambd=0.97 --epoch_ppo=80 \