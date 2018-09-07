from shutil import rmtree

from pathlib import Path

from time import time
from datetime import timedelta


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
    r"""Run the given experiment in parallel (master-worker) for all defined configurations. 
    
    Args:
        worker_class (BaseExperimentWorker): user-defined experiment worker class.
        master_class (BaseExperimentMaster): user-defined experiment master class.
        max_num_worker (int, optional): maximum number of workers. It has the following use cases:
            * If ``None``, then the number of wokers set to be the total number of configurations. 
            * If the number of configurations is less than this max bound, then the number of 
              workers will be automatically reduced to the number of configurations.
            * If the number of configurations is larger than this max bound, then the rest of 
              configurations will be fed in iteratively as batches each with the size maximally
              equal to this bound. 
        daemonic_worker (bool, optional): If ``True``, then set each worker to be daemonic. 
            See :class:`BaseWorker` and :class:`BaseMaster` for more details about daemonic
    """
    # Set start time
    t = time()
    
    # Create experiment
    experiment = master_class(worker_class=worker_class, 
                              max_num_worker=max_num_worker, 
                              daemonic_worker=daemonic_worker)
    
    # Create path to log directory defined in the configuration
    log_path = Path(experiment.configs[0]['log:dir'])
    if not log_path.exists():  # Make directory if it does not exist
        log_path.mkdir(parents=True)  # create recursively for all missing directories
    else:  # already existed, ask user whether to remove old logs
        msg = f"Logging directory '{log_path.absolute()}' already existed, do you want to clean it ?"
        answer = ask_yes_or_no(msg)
        if answer:
            # Remove everything recursively under logging directory
            rmtree(log_path)
            # Create root logging directory
            log_path.mkdir()
        else:
            old_log_path = log_path.with_name('old_' + log_path.name)
            # Rename directory log to old_log
            log_path.rename(old_log_path)
            # Create a new directory for current directory
            log_path.mkdir()
            print(f"The old logging directory is renamed to '{old_log_path.absolute()}'. ")
            input('Please, press Enter to continue\n>>> ')
            
    # Create subfolders for each ID
    for i in range(len(experiment.configs)):  # start from 0
        p = log_path / f'{i}'
        p.mkdir()
        
    # Save all configurations
    experiment.save_configs(log_path / 'configs.npy')
            
    # Run experiment in parallel
    experiment()

    # Total time logging
    print(f'\nTotal time: {timedelta(seconds=round(time() - t))}')
