python train_PPO_lag.py --env_name=Safexp-PointGoal1-v0 --buffer_size=25000 --eval_interval=50000 \
--gamma=0.97 --reward_factor=1 --seed=1 --lr_actor=0.005 --lr_critic=0.005 --lr_cost_critic=0.005 \
 --hidden_units_actor=128 --cost_limit=25 --lr_penalty=0.1 \
--hidden_units_critic=128 --lambd=0.97 --epoch_ppo=50 \
--num_training_step=10000000 \
--wandb_logs=True \