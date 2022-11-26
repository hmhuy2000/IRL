# python train_PPO.py --env_name=CartPole-v1 --buffer_size=2500 --eval_interval=5000 \
# --gamma=0.98 --reward_factor=0.1 --seed=0 --lr_actor=0.0001 --lr_critic=0.0001 --hidden_units_actor=64 \
# --hidden_units_critic=64 --lambd=0.95 --epoch_ppo=25 --max_episode_length=500 \
# --num_training_step=1000000 \
# --wandb_logs=True

# python train_PPO.py --env_name=InvertedPendulum-v4 --buffer_size=50000 --eval_interval=100000 \
# --gamma=0.98 --reward_factor=0.1 --seed=0 --lr_actor=0.001 --lr_critic=0.001 --hidden_units_actor=64 \
# --hidden_units_critic=64 --lambd=0.95 --epoch_ppo=50 --max_episode_length=1000 \
# --num_training_step=1000000 \
# --wandb_logs=True

# python train_PPO.py --env_name=Hopper-v4 --buffer_size=25000 --eval_interval=50000 \
# --gamma=0.99 --reward_factor=0.1 --seed=0 --lr_actor=0.005 --lr_critic=0.005 --hidden_units_actor=64 \
# --hidden_units_critic=64 --lambd=0.97 --epoch_ppo=50 \
# --num_training_step=3000000 \
# --wandb_logs=True \

python train_PPO.py --env_name=Safexp-PointGoal1-v0 --buffer_size=25000 --eval_interval=50000 \
--gamma=0.97 --reward_factor=1 --seed=0 --lr_actor=0.005 --lr_critic=0.005 --hidden_units_actor=128 \
--hidden_units_critic=128 --lambd=0.97 --epoch_ppo=50 \
--num_training_step=10000000 \
--wandb_logs=True \