from itertools import product

from lagom.multiprocessing import ProcessMaster


class ExperimentMaster(ProcessMaster):
    r"""The master of parallelized experiment. 
    
    It receives a :class:`Config` object and generate a list of all possible configurations. It also receives
    a list of random seeds, and each configuration runs with each random seeds. Then it distributes each
    pair of configuration and seed to the workers. 

    """
    def __init__(self, worker_class, num_worker, run, config, seeds):
        super().__init__(worker_class, num_worker)
        
        self.run = run
        self.config = config
        self.seeds = seeds
        self.configs = self.config.make_configs()
    
    def make_tasks(self):
        tasks = list(product(self.configs, self.seeds))
        tasks = [(config, seed, self.run) for config, seed in tasks]
        return tasks
