import shutil
import tarfile
from pathlib import Path


# extract all_logs to each algorithm folder
path = Path.cwd()
assert path.name == 'baselines'
assert (path / 'all_logs.tar.gz').exists()
algos = [x for x in path.iterdir() if x.is_dir() and (x / 'experiment.py').exists()]
all_logs_path = path / 'all_logs'
if all_logs_path.exists():
    shutil.rmtree(all_logs_path)
print('Extracting all logs folders...')
with tarfile.open('all_logs.tar.gz') as tar:
    tar.extractall()
assert all_logs_path.exists()
print('Done !')
print('Moving all logs folders...')
[shutil.move((all_logs_path / algo.name / 'logs').as_posix(), algo.as_posix()) for algo in algos]
print('Done !')
shutil.rmtree(all_logs_path)
print('Clean up all_logs')
