#!/bin/bash
export SC2PATH=/home/lthpc/Desktop/Research/meta-role/RODE/3rdparty/StarCraftII
export CUDA_VISIBLE_DEVICES=1

for ((t=0; t<=0; t+=100000))
do
    echo $t
    python3 src/main.py --config=rode --env-config=sc2 with env_args.map_name=corridor626 n_role_clusters=3 role_interval=5 checkpoint_path=/home/lthpc/Desktop/Research/meta-role/RODE/results/models/rode__2021-02-23_17-33-26 save_replay=True use_tensorboard=True save_model=False verbose=True test_nepisode=24  load_step=$t env_args.max_n_enemies=36 env_args.max_n_agents=18 env_args.n_enemies_in_obs=24 env_args.n_allies_in_obs=5
    #python3 check.py
done

