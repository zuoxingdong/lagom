import matplotlib.pyplot as plt


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
