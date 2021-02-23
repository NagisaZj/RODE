CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=rode --env-config=sc2 with env_args.map_name=corridor n_role_clusters=3 role_interval=5 t_max=5050000 env_args.max_n_enemies=36 env_args.max_n_agents=18 env_args.n_enemies_in_obs=24 env_args.n_allies_in_obs=5


CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=rode --env-config=sc2 with env_args.map_name=corridor n_role_clusters=3 role_interval=5 checkpoint_path=./results/models/rode__2021-02-22_10-27-26 save_replay=True use_tensorboard=False save_model=False verbose=True test_nepisode=24 runner=episode

env_args.save_replay_prefix=test1