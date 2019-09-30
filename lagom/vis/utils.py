from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lagom.utils import pickle_load
from lagom.utils import yaml_load
from lagom.transform import interp_curves
from lagom.transform import smooth_filter


def set_ticker(ax, axis='x', num=None, KM_format=False, integer=False):
    if axis == 'x':
        axis = ax.xaxis
    elif axis == 'y':
        axis = ax.yaxis
    if num is not None:
        axis.set_major_locator(plt.MaxNLocator(num))
    if KM_format:
        def tick_formatter(x, pos):
            if abs(x) >= 0 and abs(x) < 1000:
                return int(x) if integer else x
            elif abs(x) >= 1000 and abs(x) < 1000000:
                return f'{int(x/1000)}K' if integer else f'{x/1000}K'
            elif abs(x) >= 1000000:
                return f'{int(x/1000000)}M' if integer else f'{x/1000000}M'
        axis.set_major_formatter(plt.FuncFormatter(tick_formatter))
    return ax


def read_xy(log_folder, file_name, get_x, get_y, smooth_out=False, smooth_kws=None, point_step=1):
    glob_dir = lambda x: [p for p in x.glob('*/') if p.is_dir() and str(p.name).isdigit()]
    dfs = []
    for id_folder in glob_dir(Path(log_folder)):
        x = []
        y = []
        for seed_folder in glob_dir(id_folder):
            logs = pickle_load(seed_folder / file_name)
            x.append([get_x(log) for log in logs])
            y.append([get_y(log) for log in logs])
        new_x, ys = interp_curves(x, y)  # all seeds share same x values
        
        if smooth_out:
            if smooth_kws is None:
                smooth_kws = {'window_length': 51, 'polyorder': 3}
            ys = [smooth_filter(y, **smooth_kws) for y in ys]
        
        idx = np.arange(0, new_x.size, step=point_step)
        new_x = new_x[idx, ...]
        ys = [y[idx, ...] for y in ys]

        df = pd.DataFrame({'x': np.tile(new_x, len(ys)), 'y': np.hstack(ys)})
        config = yaml_load(id_folder / 'config.yml')
        config = pd.DataFrame([config.values()], columns=config.keys())
        config = config.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
        df = pd.concat([df, config], axis=1, ignore_index=False)
        df = df.fillna(method='pad')  # padding all NaN configs
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0, ignore_index=True)    
    return dfs
