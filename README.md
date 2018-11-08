# POP3D
Source code(Tensorflow)for Policy Optimization With Penalized Point Probability Distance: An Alternative To Proximal Policy Optimization
(https://arxiv.org/pdf/1807.00442.pdf)
# Prerequisite
- gym[mujoco,atari]
- scipy
- tqdm
- joblib
- zmq
- dill 
- mpi4py 
- cloudpickle
- tensorflow>=1.4.0
- opencv-python
# Training
## If you desire to run all games(49 Atari or 7 Mujoco) using 2 GPUs( You can change the task distribution work based on GPU resource), you can do as follows.
- Atari 
```bash
python -m baselines.ppo2.run_all_atari
```
- Mujoco
```bash
python -m baselines.ppo2.run_all_mujoco
```
## If you want to train only one game, take Atari Alien using seed 10 for example
- Use PPO
```bash
python -m baselines.ppo2.run_atari --env AlienNoFrameskip-v4  --num-timesteps 10000000 --seed 10
```
- Use POP3D
```bash
python -m baselines.ppo2.run_atari --env AlienNoFrameskip-v4  --num-timesteps 10000000 --seed 10 --use-penal 1
```
# Results
You can download  results on three seeds from google drive https://drive.google.com/file/d/1c79TqWn74mHXhLjoTWaBKfKaQOsfD2hg/view?usp=sharing. 
We release it to make reproduction of this paper easy.
## Atari results(PPO, POP3D, TRPO, BASLINES)
![Atari](https://github.com/cxxgtxy/POP3D/blob/master/pop3d.png)
## Mujoco results
![Atari](https://github.com/cxxgtxy/POP3D/blob/master/mujoco.png)

# Acknowledge
Thanks to OpenAI's baselines, our code is based on https://github.com/openai/baselines.git