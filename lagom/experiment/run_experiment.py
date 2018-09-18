from shutil import rmtree

from pathlib import Path

from time import time
from datetime import timedelta

from lagom import color_str
from lagom import yaml_dump


def ask_yes_or_no(msg):
    r"""Ask user to enter yes or no to a given message. 
    
    Args:
        msg (str): a message
    """
    
    print(msg)
    
    while True:
        answer = str(input('>>> ')).lower().strip()
        
        if answer[0] == 'y':
            return True
        elif answer[0] == 'n':
            return False
        else:
            print("Please answer 'yes' or 'no':")


def run_experiment(worker_class, master_class, max_num_worker=None, daemonic_worker=None):
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
        max_num_worker (int, optional): See docstring in :class:`BaseExperimentMaster` for more details. 
        daemonic_worker (bool, optional): See docstring in :class:`BaseExperimentMaster` for more details. 
    """
    # Set start time
    t = time()
    
    # Create experiment
    experiment = master_class(worker_class=worker_class, 
                              max_num_worker=max_num_worker, 
                              daemonic_worker=daemonic_worker)
    
    # Create path to log directory defined in the configuration
    log_path = Path(experiment.configs[0]['log.dir'])
    if not log_path.exists():  # Make directory if it does not exist
        log_path.mkdir(parents=True)  # create recursively for all missing directories
    else:  # already existed, ask user whether to remove old logs
        msg = f"Logging directory '{log_path.absolute()}' already existed, do you want to clean it ?"
        answer = ask_yes_or_no(msg)
        if answer:
            # Remove everything recursively under logging directory
            rmtree(log_path)
            # Create root logging directory
            log_path.mkdir(parents=True)
        else:  # back up
            old_log_path = log_path.with_name('old_' + log_path.name)
            # Rename directory log to old_log
            log_path.rename(old_log_path)
            # Create a new directory for current directory
            log_path.mkdir(parents=True)
            print(f"The old logging directory is renamed to '{old_log_path.absolute()}'. ")
            input('Please, press Enter to continue\n>>> ')
            
    # Create subfolders for each ID and subsubfolders for each random seed
    for config in experiment.configs:
        ID = config['ID']
        # Create folders recursively
        for seed in experiment.seeds:  # iterate over all seeds
            p = log_path / f'{ID}' / f'{seed}'
            p.mkdir(parents=True)
        # Yaml dump configuration dictionary
        yaml_dump(obj=config, f=log_path/f'{ID}'/'config', ext='.yml')

    # Save all configurations
    experiment.save_configs(log_path / 'configs')
            
    # Run experiment in parallel
    experiment()

    # Total time logging
    msg = color_str(f'\nTotal time: {timedelta(seconds=round(time() - t))}', 'green', 'bold')
    print(msg)
