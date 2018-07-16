import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

from .base_plot import BasePlot


class CurvePlot(BasePlot):
    """
    Compare different curves in one plot. In machine learning research, it is 
    extremely useful, e.g. compare training losses with different baselines. 
    
    Note that the uncertainty (error bands) is supported, a standard use case
    is that each baseline run several times by using different random seeds. 
    
    Either with or without uncertainty, it depends on what kind of data added
    to the plotter via `add(name, data)`. If the data is one-dimentional, it will
    be treated as a single curve. If the data is two-dimensional, it will be plotted
    with uncertainty. 
    
    To generate a modern high quality research plot, we use Seaborn.lineplot with 
    Pandas.DataFrame data structure. 
    
    For more advanced use cases, feel free to inherit this class and overide `__call__`. 
    """
    def __call__(self, colors=None, scales=None, alphas=None, **kwargs):
        """
        Args:
            colors (list): A list of colors, each for plotting one data item
            scales (list): A list of scales of standard deviation, each for plotting one uncertainty band
            alphas (list): A list of alphas (transparency), each for plotting one uncertainty band
            **kwargs: keyword aguments used to specify the plotting options. 
                 A list of possible options:
                     - title (str): title of the plot
                     - xlabel (str): label of horizontal axis
                     - ylabel (str): label of vertical axis
                     - xlim (tuple): [min, max] as limit of horizontal axis
                     - ylim (tuple): [min, max] as limit of veritical axis
                     - logx (bool): log-scale of horizontal axis
                     - logy (bool): log-scale of vertical axis
                     - integer_x (bool): enforce integer coordinates of horizontal axis
                     - integer_y (bool): enforce integer coordinates of vertical axis
                     
            
        Returns:
            ax (Axes): A matplotlib Axes representing the generated plot.
        """
        # Use seaborn to make figure looks modern
        sns.set()
        
        # Seaborn auto-coloring
        if colors is None:
            colors = sns.color_palette(n_colors=len(self.data))
        
        # Create a figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6, 4])
        
        # Iterate all saved data
        for i, (name, data) in enumerate(self.data.items()):
            # Enforce ndarray data with two-dimensional
            data = self._make_data(data)
            # Retrieve the generated color
            color = colors[i]
            
            # Compute the mean of the data
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            # Number of values
            N = mean.size
            
            # Plot mean curve
            ax.plot(range(1, N+1), 
                    mean, 
                    color=color, 
                    label=name)
            # Plot all uncertainty bands
            if scales is None or alphas is None:  # if nothing provided, one std uncertainty band by default
                scales = [1.0]
                alphas = [0.5]
            for scale, alpha in zip(scales, alphas):
                ax.fill_between(range(1, N+1), 
                                mean - scale*std, 
                                mean + scale*std, 
                                facecolor=color, 
                                alpha=alpha)
                
        # Make legend for each mean curve (data item)
        ax.legend()
        
        # Make title if provided
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        
        # Make x-y label if provided
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'])
        
        # Enforce min/max of axes if provided
        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'])
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'])
            
        # Enforce log-scale of axes if provided
        if 'logx' in kwargs and kwargs['logx']:
            ax.set_xscale('log')
        if 'logy' in kwargs and kwargs['logy']:
            ax.set_yscale('log')
        
        # Enforce axis having integer coordinates if provided
        if 'integer_x' in kwargs and kwargs['integer_x']:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if 'integer_y' in kwargs and kwargs['integer_y']:
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))    
        
        return ax
    
    def _make_data(self, x):
        # Enforce the data being ndarray
        x = np.array(x)
        # Enforce 2-dim data
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        elif x.ndim == 2:
            x = x
        else:
            raise TypeError(f'The input data must be either one- or two-dimensional. But got {all_data.ndim}.')
            
        return x