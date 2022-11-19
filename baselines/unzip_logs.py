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
    
    import os
    
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar)
assert all_logs_path.exists()
print('Done !')
print('Moving all logs folders...')
[shutil.move((all_logs_path / algo.name / 'logs').as_posix(), algo.as_posix()) for algo in algos]
print('Done !')
shutil.rmtree(all_logs_path)
print('Clean up all_logs')
