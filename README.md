# POP3D
Source code for Policy Optimization with Penalized Point Probability Distance: an Alternative to Proximal Policy Optimization 
# Training
## If you desire to run all games once, you can do as follows.
- Atari 
```python
python -m baselines.ppo2.run_all_atari
```
- Mujoco
```python
python -m baselines.ppo2.run_all_mujoco
```
## If you want to train only one game, take Atari Alien using seed 10 for example
- Use PPO
```python
python -m baselines.ppo2.run_atari --env AlienNoFrameskip-v4  --num-timesteps 10000000 --seed 10
```
- Use POP3D
```python
python -m baselines.ppo2.run_atari --env AlienNoFrameskip-v4  --num-timesteps 10000000 --seed 10 --use-penal 1
```
# Results
## Atari results
![Atari](https://github.com/cxxgtxy/POP3D/blob/master/pop3d.png)
## Mujoco reults
![Atari](https://github.com/cxxgtxy/POP3D/blob/master/mujoco.png)

# Acknowledge
Thanks to OpenAI's baselines, our code is based on https://github.com/openai/baselines.git