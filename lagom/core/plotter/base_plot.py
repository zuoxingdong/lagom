class BasePlot(object):
    r"""Base class for plotting the experiment result. 
    
    Many modern research plots are done via Seaborn, exploiting DataFrame data structure
    from Pandas e.g.:
    
    * loss curves with uncertainties from different random seeds. 
    
    * Heatmaps with value shown in each cell. 
    
    * Kernel Density Estimation (KDE) plots.
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    
    """
    def __init__(self):
        self.data = {}
    
    def add(self, name, data):
        r"""Add a new data for plotting. 
        
        Args:
            name (str): name of the given data
            data (object): given data
        """
        self.data[name] = data
        
    def __call__(self, **kwargs):
        r"""Generate a plot. 
        
        Args:
            **kwargs: keyword aguments used to specify the plotting options. 
            
        Returns
        -------
        ax : Axes
            a matplotlib Axes representing the generated plot. 
        """
        raise NotImplementedError
