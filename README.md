# HumanPlus: Humanoid Shadowing and Imitation from Humans


#### Project Website: https://humanoid-ai.github.io/

This repository contains the updating implementation for the Humanoid Shadowing Transformer (HST) and the Humanoid Imitation Transformer (HIT), along with instructions for whole-body pose estimation and the associated hardware codebase.


## Humanoid Shadowing Transformer (HST)
Reinforcement learning in simulation is based on [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl).
#### Installation
Install IsaacGym v4 first from the [official source](https://developer.nvidia.com/isaac-gym). Place the isaacgym fold inside the HST folder.

    conda create -n HST python=3.8(也可以先不加python 安装) 
    conda activate HST
    cd HST/rsl_rl && pip install -e . 
    cd HST/legged_gym && pip install -e .

 #pay attention
#如果找不到虚拟环境中的python，则：

`export LD_LIBRARY_PATH=/home/tommy/anaconda3/envs/HSTHIT/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`

#运行RL训练前额外需要安装的库：

```
pip install wandb
pip install ipython
pip install Pillow==9.0.0 #更改版本
```

修改强化学习训练过程中train.py中的wandb账号用户名、工程项目名：
```
def train(args):
    wandb.init(project='humanoid', name=args.run_name, entity="wandb名字")
```    

#### Example Usages
To train HST:
    如果第一次训练过程中被中断掉，第二次运行训练无法成功，主要是GPU被占用，需要重启或杀掉相应的进程    

    python legged_gym/scripts/train.py --run_name 0001_test --headless --sim_device cuda:0 --rl_device cuda:0

To play a trained policy:

    python legged_gym/scripts/play.py --load_run 0001_test --checkpoint -1 --headless --sim_device cuda:0 --rl_device cuda:0


## Humanoid Imitation Transformer (HIT)
Imitation learning in the real world is based on [ACT repo](https://github.com/tonyzhaozh/act) and [Mobile ALOHA repo](https://github.com/MarkFzp/act-plus-plus).
#### Installation
    conda create -n HIT python=3.8.10
    conda activate HIT
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    pip install getkey
    pip install wandb
    pip install chardet
    pip install h5py_cache
    cd HIT/detr && pip install -e .
#### Example Usages
Collect your own data or download our dataset from [here](https://drive.google.com/drive/folders/1i3eGTd9Nl_tSieoE0grxuKqUAumBr2EV?usp=drive_link) and place it in the HIT folder.

To set up a new terminal, run:

    conda activate HIT
    cd HIT

To train HIT:

    # Fold Clothes task
    python imitate_episodes_h1_train.py --task_name data_fold_clothes --ckpt_dir fold_clothes/ --policy_class HIT --chunk_size 50 --hidden_dim 512 --batch_size 48 --dim_feedforward 512 --lr 1e-5 --seed 0 --num_steps 100000 --eval_every 100000 --validate_every 1000 --save_every 10000 --no_encoder --backbone resnet18 --same_backbones --use_pos_embd_image 1 --use_pos_embd_action 1 --dec_layers 6 --gpu_id 0 --feature_loss_weight 0.005 --use_mask --data_aug --wandb

## Hardware Codebase
Hardware codebase is based on [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2).

#### Installation

install [unitree_sdk](https://github.com/unitreerobotics/unitree_sdk2)

install [unitree_ros2](https://support.unitree.com/home/en/developer/ROS2_service)

    conda create -n lowlevel python=3.8
    conda activate lowlevel

install [nvidia-jetpack](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-461/install-jetpack/index.html)

install torch==1.11.0 and torchvision==0.12.0:  
please refer to the following links:   
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

#### Example Usages
Put your trained policy in the `hardware-script/ckpt` folder and rename it to `policy.pt`

    conda activate lowlevel
    cd hardware-script
    python hardware_whole_body.py --task_name stand


## Pose Estimation
For body pose estimation, please refer to [WHAM](https://github.com/yohanshin/WHAM). 
For hand pose estimation, please refer to [HaMeR](https://github.com/geopavlakos/hamer). 
