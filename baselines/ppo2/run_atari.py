import sys
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
import multiprocessing
import tensorflow as tf
import os


def train(env_id, num_timesteps, seed, policy, args):

    #ncpu = multiprocessing.cpu_count()
    ncpu = 3
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)
    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'mlp': MlpPolicy}[policy]
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=3, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1), args=args)

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    parser.add_argument('--use-penal', help='enable penal', default=False)
    parser.add_argument('--gpu', type=int, default=0, help='GPU selection')
    parser.add_argument('--pg-rate', type=float, default=0.0)
    # parser.add_argument('--save-dir', default='.logs', type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % args.gpu
    print("game %s run on GPU: %d" % (args.env, args.gpu))
    logger.configure(args.env + '_seed_' + str(args.seed) + '_nopen' + '_pg' + str(args.pg_rate) if not args.use_penal
                     else args.env + '_seed_' + str(args.seed) + '_pen' + '_pg' + str(args.pg_rate), ['log', 'tensorboard'])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, args=args)

if __name__ == '__main__':
    main()


