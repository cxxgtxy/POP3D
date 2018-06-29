from multiprocessing import Process
import subprocess
import time
# noinspection PyPackageRequirements
envs=[
'Alien',
'Amidar',
'Assault',
'Asterix',
'Asteroids',
'Atlantis',
'BankHeist',
'BattleZone',
'BeamRider',
'Bowling',
'Boxing',
'Breakout',
'Centipede',
'ChopperCommand',
'CrazyClimber',
'DemonAttack',
'DoubleDunk',
'Enduro',
'FishingDerby',
'Freeway',
'Frostbite',
'Gopher',
'Gravitar',
'IceHockey',
'Jamesbond',
'Kangaroo',
'Krull',
'KungFuMaster',
'MontezumaRevenge',
'MsPacman',
'NameThisGame',
'Pitfall',
'Pong',
'PrivateEye',
'Qbert',
'Riverraid',
'RoadRunner',
'Robotank',
'Seaquest',
'SpaceInvaders',
'StarGunner',
'Tennis',
'TimePilot',
'Tutankham',
'UpNDown',
'Venture',
'VideoPinball',
'WizardOfWor',
'Zaxxon'
]
seeds = [10, 100, 1000]


class MyProcess(Process):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        super(MyProcess, self).__init__(group=group, target=target, name=name, args=args, kwargs=kwargs)
        self._target = target
        self._args = args
        self._kwargs= kwargs

    def run(self):
        try:
            self._target(*self._args, **self._kwargs)
        except subprocess.CalledProcessError as err:
            print("error", err)

    
def call_once(game, seed, use_penal, device, timesteps, pgcoef):
    if use_penal:
        return MyProcess(target=subprocess.check_call, args=(['python', '-m', 'baselines.ppo2.run_atari', '--env', game, '--seed', str(seed),
                                         '--use-penal', str(1), '--gpu', str(device), '--num-timesteps', str(timesteps), '--pg-rate', str(pgcoef) ],))
    else:
        return MyProcess(target=subprocess.check_call, args=(['python', '-m', 'baselines.ppo2.run_atari', '--env', game, '--seed', str(seed),
                                         '--gpu', str(device), '--num-timesteps', str(timesteps), '--pg-rate', str(pgcoef)],))


def run_all():
    pro_list = list()
    steps = int(10e6)
    for seed in seeds:
        for index,env in enumerate(envs):
            pro_list.append(call_once(env+'NoFrameskip-v4', seed, False, 0, steps, 0))
            pro_list.append(call_once(env+'NoFrameskip-v4', seed, True, 1, steps, 0))

    for p in pro_list:
        p.daemon = True
    jobs = len(pro_list)
    print("total jobs:", jobs)
    start_time = time.time()
    while pro_list:
        local_ps = list()
        while len(local_ps) < 24:
            if not len(pro_list): break
            local_ps.append(pro_list.pop(0))
        print("local ps len:", len(local_ps))
        for p in local_ps:
            p.start()
        for p in local_ps:
            p.join()
        print("processing percent ------- %f" % (1-len(pro_list)/jobs))
        print("use time:", time.time()-start_time)
if __name__ == '__main__':
    run_all()



