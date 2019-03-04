from itertools import chain
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns

from lagom.transform import interp_curves


def lineplot(ax, x, y, **kwargs):
    r"""A wrapper of Seaborn `lineplot` function to support uncertainty plot for a batch of curves
    with inconsistent x-values. 
    
    Example:
    
        >>> x1 = [23, 40, 50, 60, 90, 120]
        >>> y1 = [0.5, 12.5, 15.5, 16.5, 13.4, 19]
        >>> x2 = [30, 50, 70, 90, 110]
        >>> y2 = [2.5, 18.4, 19.6, 22.3, 26]

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)
        >>> ax = lineplot(ax, [x1, x2], [y1, y2])
        >>> plt.plot(x1, y1, 'red')
        >>> plt.plot(x2, y2, 'green')
    
    """
    sns.set()
    x, y = interp_curves(x, y, num_point=200)
    ax = sns.lineplot(x=list(chain.from_iterable(x)), y=list(chain.from_iterable(y)), ax=ax, **kwargs)
    return ax


def auto_ax(ax, 
            title=None, 
            xlabel=None, 
            ylabel=None, 
            xlim=None, 
            ylim=None, 
            logx=False, 
            logy=False, 
            intx=False, 
            inty=False, 
            num_tick=None,
            xmagnitude=None, 
            legend_kws=None):
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if intx:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if inty:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if num_tick is not None:
        ax.xaxis.set_major_locator(plt.MaxNLocator(num_tick))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(partial(tick_formatter, scale_magnitude=xmagnitude)))
    if legend_kws is not None:
        ax.legend(**legend_kws)
    else:
        ax.get_legend().set_visible(False)
    
    return ax


def tick_formatter(x, pos, scale_magnitude=None):
    r"""A function to set major functional formatter. 

    Args:
        x (object): data value, internal argument used by Matplotlib
        pos (object): position, internal argument used by Matplotlib
        scale_magnitude (str): string description of scaling magnitude, use functools.partial
            to make a function specified with this scaler but without put it as a required argument. 
            Possible values: 

                - 'N': no scaling, raw value
                - 'K': every one thousand
                - 'M': every one million

            When None is given, then it automatically detect for N, K or M. 

    Returns
    -------
    out : str
        a formatted string of the tick given the data value. 
    """
    msg = f'expected K, M or None, got {scale_magnitude}'
    assert scale_magnitude in ['N', 'K', 'M', None], msg

    # Auto-assign scale magnitude to K or M depending on the x value
    if scale_magnitude is None:
        if x < 1000:  # less than a thousand, so show raw value
            scale_magnitude = 'N'
        elif x >= 1000 and x < 1000000:  # between a thousand and a million
            scale_magnitude = 'K'
        elif x >= 1000000:  # more than a million
            scale_magnitude = 'M'

    # Make scaler based on string format
    if scale_magnitude == 'N':
        scaled_x = x
        scale_str = ''
    elif scale_magnitude == 'K':
        scaled_x = x/1000
        scale_str = 'K'
    elif scale_magnitude == 'M':
        scaled_x = x/1000000
        scale_str = 'M'

    # Format the data value and return it
    if scaled_x == 0:  # no format for zero
        return '0'
    elif scaled_x > 0 and scaled_x < 1:  # less than 1, then keep one decimal
        return f'{round(scaled_x, 1)}{scale_str}'
    else:  # more than 1
        return f'{int(scaled_x)}{scale_str}'
