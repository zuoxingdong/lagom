# TEMP: Import lagom
# Not useful once lagom is installed
import sys
sys.path.append('/home/zuo/Code/lagom/')

from time import time

from experiment import Experiment
from algo import Algorithm


algo = Algorithm(name='test')
experiment = Experiment()

experiment.add_algo(algo)


start_time = time()

experiment.benchmark(num_process=1)
print(f'\nTotal time: {time() - start_time:.2f} s')

# Save all configurations
experiment.save_configs()