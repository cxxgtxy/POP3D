# POP3D
Source code for Policy Optimization With Penalized Point Probability Distance: An Alternative To Proximal Policy Optimization
(https://arxiv.org/abs/1807.00442)
# Training
## If you desire to run all games once, you can do as follows.
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
## Atari results
![Atari](https://github.com/cxxgtxy/POP3D/blob/master/pop3d.png)
## Mujoco reults
![Atari](https://github.com/cxxgtxy/POP3D/blob/master/mujoco.png)

# Acknowledge
Thanks to OpenAI's baselines, our code is based on https://github.com/openai/baselines.git