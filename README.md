# LLM 4 Materials for VIP @ GT

## Installation
```
# 1. request a GPU on ICE
# e.g. request a single V100
salloc -N1 -t02:00:00 --gres=gpu:V100:1  --ntasks-per-node=4

# 2. load conda package manager
module load anaconda3

# create environment
conda create -n llmft python=3.11

# install torch
pip install torch torchvision torchao

# install torchtune
git clone https://github.com/pytorch/torchtune.git
pip install -e torchtune/

# install this package
git clone https://github.com/shuyijia/llm4mat-vip.git
pip install -e llm4mat-vip

# install any additional packages you might need
pip install ase
```