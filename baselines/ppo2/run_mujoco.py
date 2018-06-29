#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
import os


def train(env_id, num_timesteps, seed, args):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=64,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        args=args)


def main():
    parser = mujoco_arg_parser()
    parser.add_argument('--use-penal', help='enable penal', default=False)
    parser.add_argument('--gpu', type=int, default=0, help='GPU selection')
    parser.add_argument('--pg-rate', type=float, default=0.0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % args.gpu
    print("game %s run on GPU: %d" % (args.env, args.gpu))
    logger.configure(args.env + '_seed_' + str(args.seed) + '_nopen' + '_pg' + str(args.pg_rate) if not args.use_penal
                     else args.env + '_seed_' + str(args.seed) + '_pen' + '_pg' + str(args.pg_rate), ['log', 'tensorboard'])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, args=args)


if __name__ == '__main__':
    main()

