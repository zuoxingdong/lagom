from shutil import rmtree
from pathlib import Path

from lagom.utils import pickle_dump
from lagom.utils import yaml_dump
from lagom.utils import ask_yes_or_no
from lagom.utils import timeit

from .experiment_master import ExperimentMaster
from .experiment_worker import ExperimentWorker


@timeit(color='green', attribute='bold')
def run_experiment(run, config, seeds, num_worker):
    r"""A convenient function to launch a parallelized experiment (Master-Worker). 
    
    .. note::
    
        It automatically creates all subfolders for logging the experiment. The topmost
        folder is indicated by the logging directory specified in the configuration. 
        Then all subfolders for each configuration are created with the name of their ID.
        Finally, under each configuration subfolder, a set subfolders are created for each
        random seed (the random seed as folder name). Intuitively, an experiment could have 
        following directory structure::
        
            - logs
                - 0  # ID number
                    - 123  # random seed
                    - 345
                    - 567
                - 1
                    - 123
                    - 345
                    - 567
                - 2
                    - 123
                    - 345
                    - 567
                - 3
                    - 123
                    - 345
                    - 567
                - 4
                    - 123
                    - 345
                    - 567
    
    Args:
        run (function): an algorithm function to train on.
        config (Config): a :class:`Config` object defining all configuration settings
        seeds (list): a list of random seeds
        num_worker (int): number of workers
        
    """
    experiment = ExperimentMaster(ExperimentWorker, num_worker, run, config, seeds)
    
    log_path = Path(experiment.configs[0]['log.dir'])
    if not log_path.exists():
        log_path.mkdir(parents=True)
    else:
        msg = f"Logging directory '{log_path.absolute()}' already existed, do you want to clean it ?"
        answer = ask_yes_or_no(msg)
        if answer:
            rmtree(log_path)
            log_path.mkdir(parents=True)
        else:  # back up
            old_log_path = log_path.with_name('old_' + log_path.name)
            log_path.rename(old_log_path)
            log_path.mkdir(parents=True)
            print(f"The old logging directory is renamed to '{old_log_path.absolute()}'. ")
            input('Please, press Enter to continue\n>>> ')
            
    # Create subfolders for each ID and subsubfolders for each random seed
    for config in experiment.configs:
        ID = config['ID']
        for seed in experiment.seeds:
            p = log_path / f'{ID}' / f'{seed}'
            p.mkdir(parents=True)
        yaml_dump(obj=config, f=log_path/f'{ID}'/'config', ext='.yml')

    pickle_dump(experiment.configs, log_path / 'configs', ext='.pkl')
            
    # Run experiment in parallel
    results = experiment()
    return results
