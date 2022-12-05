# python collect_data.py --env_name=InvertedPendulum-v4 --buffer_size=1000000 \
# --seed=0 --hidden_units_actor=64 --hidden_units_critic=64 --max_episode_length=1000 \
# --load_dir='./weights/InvertedPendulum-v4-9' \
# --buffer_dir='./buffers'

# python collect_data.py --env_name=Hopper-v4 --buffer_size=100000 \
# --seed=0 --hidden_units_actor=64 --hidden_units_critic=64 --max_episode_length=1000 \
# --load_dir='./weights/PPO/Hopper-v4-3720.88' \
# --buffer_dir='./buffers'

# python collect_data.py --env_name=CartPole-v1 --buffer_size=100000 \
# --seed=0 --hidden_units_actor=64 --hidden_units_critic=64 --max_episode_length=500 \
# --load_dir='./weights/PPO/CartPole-v1-500.00' \
# --buffer_dir='./buffers'

python collect_data.py --env_name=Safexp-PointGoal1-v0 --buffer_size=5000 \
--seed=0 --hidden_units_actor=128 --hidden_units_critic=128 \
--load_dir='./weights/PPO/Safexp-PointGoal1-v0-26.48' \
--buffer_dir='./buffers'