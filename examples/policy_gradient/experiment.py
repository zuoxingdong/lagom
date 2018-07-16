from algo import Algorithm

from lagom.experiment import Config
from lagom.experiment import BaseExperimentWorker
from lagom.experiment import BaseExperimentMaster


class ExperimentWorker(BaseExperimentWorker):
    def make_algo(self):
        algo = Algorithm(name='Policy gradient in CartPole')
        
        return algo


class ExperimentMaster(BaseExperimentMaster):
    def process_algo_result(self, config, result):
        assert result == None
        
    def make_configs(self):
        config = Config()
        
        config.add_grid(name='cuda', val=[True])
        config.add_grid(name='seed', val=[834748646,  635037080,  984275692,  225168546,  713396787])
        
        config.add_item(name='lr', val=1e-2)
        config.add_item(name='gamma', val=0.995)
        config.add_item(name='standardize_r', val=True)
        config.add_item(name='train_iter', val=1000)
        config.add_item(name='N', val=1)
        config.add_item(name='T', val=300)
        
        config.add_item(name='use_value', val=False)  # True for actor-critic
        config.add_item(name='entropy_coef', val=0.0)
        config.add_item(name='value_coef', val=0.0)
        
        config.add_item(name='max_grad_norm', val=None)
        
        config.add_item(name='log_interval', val=100)
        
        configs = config.make_configs()
        
        return configs