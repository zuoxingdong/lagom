from shutil import rmtree

from pathlib import Path

from time import time
from datetime import timedelta

from lagom.utils import color_str
from lagom.utils import yaml_dump
from lagom.utils import ask_yes_or_no


def run_experiment(worker_class, master_class, num_worker):
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
        worker_class (BaseExperimentWorker): a worker class for the experiment.
        master_class (BaseExperimentMaster): a master class for the experiment.
        num_worker (int, optional): number of workers. 
    """
    t = time()
    
    experiment = master_class(worker_class=worker_class, num_worker=num_worker)
    
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

    experiment.save_configs(log_path / 'configs')
            
    # Run experiment in parallel
    experiment()
    print(color_str(f'\nTotal time: {timedelta(seconds=round(time() - t))}', 'green', 'bold'))
