from multiprocessing import Process
import subprocess
import time

# noinspection PyPackageRequirements
envs = [
    'HalfCheetah-v2',
    'Hopper-v2',
    'InvertedDoublePendulum-v2',
    'InvertedPendulum-v2',
    'Reacher-v2',
    'Swimmer-v2',
    'Walker2d-v2'
]
seeds = [0, 10, 100]


class MyProcess(Process):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        super(MyProcess, self).__init__(group=group, target=target, name=name, args=args, kwargs=kwargs)
        self._target = target
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self._target(*self._args, **self._kwargs)
        except subprocess.CalledProcessError as err:
            print("error", err)


def call_once(game, seed, use_penal, device, timesteps, pgcoef):
    if use_penal:
        return MyProcess(target=subprocess.check_call,
                         args=(['python', '-m', 'baselines.ppo2.run_mujoco', '--env', game, '--seed', str(seed),
                                '--use-penal', str(1), '--gpu', str(device), '--num-timesteps', str(timesteps),
                                '--pg-rate', str(pgcoef)],))
    else:
        return MyProcess(target=subprocess.check_call,
                         args=(['python', '-m', 'baselines.ppo2.run_mujoco', '--env', game, '--seed', str(seed),
                                '--gpu', str(device), '--num-timesteps', str(timesteps), '--pg-rate', str(pgcoef)],))


def run_all():
    # counter = 0
    pro_list = list()
    steps = int(1e7)
    for seed in seeds:
        for index, env in enumerate(envs):
            pro_list.append(call_once(env, seed, False, 0, steps, 0))
            pro_list.append(call_once(env, seed, True, 1, steps, 0.0))

    for p in pro_list:
        p.daemon = True
    jobs = len(pro_list)
    print("total jobs:", jobs)
    start_time = time.time()
    while pro_list:
        local_ps = list()
        while len(local_ps) < 12:
            if not len(pro_list): break
            local_ps.append(pro_list.pop(0))
        print("local ps len:", len(local_ps))
        for p in local_ps:
            p.start()
        for p in local_ps:
            p.join()
        print("processing percent ------- %f" % (1 - len(pro_list) / jobs))
        print("use time:", time.time() - start_time)


if __name__ == '__main__':
    run_all()



