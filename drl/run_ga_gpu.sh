#!/bin/bash

if [ -z "$1" ]
  then echo "Please provide the name of the game, e.g.  ./run_cpu breakout "; exit 0
fi
ENV=$1
FRAMEWORK="alewrap"

game_path=$PWD"/roms/"
env_params="useRGB=true"
agent="NeuralQLearner"
n_replay=1
netfile="\"ga_convnet\""
update_freq=4
actrep=4
discount=0.99
seed=1
learn_start=50000
pool_frms_type="\"max\""
pool_frms_size=2
initial_priority="false"
replay_memory=1000000
eps_end=0.1
eps_endt=replay_memory
lr=0.00025
agent_type="DQN3_0_1"
preproc_net="\"net_downsample_2x_full_y\""
agent_name=$agent_type"_"$1"_FULL_Y"
state_dim=7056
ncols=1
steps=50000
population=20
elite_size=1
random_starts=30
pool_frms="type="$pool_frms_type",size="$pool_frms_size
num_threads=3
gpu=0

args="-framework $FRAMEWORK -game_path $game_path -env $ENV -steps $steps -population $population -elite_size $elite_size -threads $num_threads -gpu $gpu"
echo $args

cd ga
../torch/bin/luajit train_ga.lua $args
