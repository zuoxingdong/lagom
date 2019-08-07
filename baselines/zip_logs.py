import shutil
import tarfile
from pathlib import Path


# move logs folder for each algorithm to a common all_logs folder and zip it
path = Path.cwd()
assert path.name == 'baselines'
algos = [x for x in path.iterdir() if x.is_dir() and (x / 'logs').exists()]
all_logs_path = path / 'all_logs'
if all_logs_path.exists():
    shutil.rmtree(all_logs_path)
if (path / 'all_logs.tar.gz').exists():
    (path / 'all_logs.tar.gz').unlink()
all_logs_path.mkdir(parents=True)
[(all_logs_path / algo.name).mkdir(parents=True) for algo in algos]
print('Moving all logs folders...')
[shutil.move((algo / 'logs').as_posix(), (all_logs_path / algo.name).as_posix()) for algo in algos]
print('Done !')
print('Creating tar.gz file...')
with tarfile.open('all_logs.tar.gz', 'x:gz') as tar:
    tar.add(all_logs_path.name)
print('Done !')
shutil.rmtree(all_logs_path)
print('Clean up all_logs folder')
