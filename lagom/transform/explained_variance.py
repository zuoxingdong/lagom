import numpy as np
from sklearn.metrics import explained_variance_score


def explained_variance(y_true, y_pred, **kwargs):
    r"""Computes the explained variance regression score.
    
    It involves a fraction of variance that the prediction explains about the ground truth.
   
    Let :math:`\hat{y}` be the predicted output and let :math:`y` be the ground truth output. Then the explained
    variance is estimated as follows:
   
    .. math::
        \text{EV}(y, \hat{y}) = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}
   
    The best score is :math:`1.0`, and lower values are worse. A detailed interpretation is as following:
   
    * :math:`\text{EV} = 1`: perfect prediction
    * :math:`\text{EV} = 0`: might as well have predicted zero
    * :math:`\text{EV} < 0`: worse than just predicting zero
   
    .. note::
    
        It calls the function from ``scikit-learn`` which handles exceptions better e.g. zero division..
        
    Example:
    
        >>> explained_variance(y_true=[3, -0.5, 2, 7], y_pred=[2.5, 0.0, 2, 8])
        0.9571734666824341
        
        >>> explained_variance(y_true=[[3, -0.5, 2, 7]], y_pred=[[2.5, 0.0, 2, 8]])
        0.9571734666824341
        
        >>> explained_variance(y_true=[[0.5, 1], [-1, 1], [7, -6]], y_pred=[[0, 2], [-1, 2], [8, -5]])
        0.9838709533214569
        
        >>> explained_variance(y_true=[[0.5, 1], [-1, 10], [7, -6]], y_pred=[[0, 2], [-1, 0.00005], [8, -5]])
        0.6704022586345673
        
    Args:
        y_true (list): ground truth output
        y_pred (list): predicted output
        **kwargs: keyword arguments to specify the estimation of the explained variance. 
           
    Returns
    -------
    out : float
        estimated explained variance
        
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    assert y_true.shape == y_pred.shape
    return explained_variance_score(y_true=y_true, y_pred=y_pred, **kwargs)
