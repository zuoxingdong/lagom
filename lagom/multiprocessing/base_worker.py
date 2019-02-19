from abc import ABC
from abc import abstractmethod


class BaseWorker(ABC):
    r"""Base class for the worker in master-worker architecture which receives 
    tasks from master and send back the result. 
    """ 
    @abstractmethod
    def work(self, task_id, task):
        r"""Work on the given task and return the result. 
        
        Args:
            task_id (int): the task ID.
            task (object): a given task. 
            
        Returns
        -------
        result : object
            working result. 
        """
        pass
