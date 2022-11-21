# python train_imitation.py --env_name=InvertedPendulum-v4 --buffer_size=2000 --eval_interval=5000 \
# --gamma=0.98 --reward_factor=0.1 --seed=0 --lr_actor=0.001 --lr_critic=0.001 --hidden_units_actor=64 \
# --hidden_units_critic=64 --lambd=0.95 --epoch_ppo=50 --max_episode_length=1000 \
# --num_training_step=10000000 --epoch_disc=5 --lr_disc=0.0001 \
# --buffer_dir='./buffers/InvertedPendulum-v4/1000000.pth' \
# --wandb_logs=True

python train_imitation.py --env_name=Hopper-v4 --buffer_size=25000 --eval_interval=50000 \
--gamma=0.99 --reward_factor=0.1 --seed=0 --lr_actor=0.005 --lr_critic=0.005 --hidden_units_actor=64 \
--hidden_units_critic=64 --lambd=0.97 --epoch_ppo=50 \
--num_training_step=3000000 --epoch_disc=5 --lr_disc=0.0001 \
--buffer_dir='./buffers/Hopper-v4/100000.pth' \
--wandb_logs=True 