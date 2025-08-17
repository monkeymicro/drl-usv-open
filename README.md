
online and offline reinforcement learning algorithm for unmanned surface vessels control

# Installation

- Ubuntu 24.04 (desktop or WSL2)

```
git clone https://github.com/drl-usv
cd drl-usv
conda create -n drl python=3.10
conda activate drl
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install scipy matplotlib pyrallis tqdm pygame
```

# Test

## manual operation

easy to change the model's parameters in usv_simulator/dynamics_simulation/boat.ini

review motion after modified.

##

online train.

run simulator

```
cd usv
python simulator-drl.py --port=11023
```

run drl algorithm.
```
cd online
python sacn.py --port=11023
```

offline train: Using online/sac.py to produce fixed-dataset.


with eval.
```
cd usv
python simulator-drl.py --port=11023
```

```
cd offline
python sacn.py --port=11023
```
