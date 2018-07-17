from time import time


def run_experiment(worker_class, master_class, num_worker, daemonic_worker=None):
    """
    This function runs given experiments in parallel (master-worker) with defined configurations. 
    
    Args:
        worker_class (BaseExperimentWorker): user-defined experiment worker class.
        master_class (BaseExperimentMaster): user-defined experiment master class.
        num_worker (int): Number of workers, each opens an Process. Recommanded to be 
            the number of available CPU core, however, it is not strictly necessary. 
        daemonic_worker (bool): If True, then set each worker to be daemonic. 
            For details of daemonic worker, please refer to documentations in classes of 
            BaseWorker/BaseMaster
    """
    t = time()

    experiment = master_class(worker_class=worker_class, 
                              num_worker=num_worker, 
                              daemonic_worker=daemonic_worker)
    # Run experiment in parallel
    experiment()

    print(f'\nTotal time: {time() - t:.2} s')
