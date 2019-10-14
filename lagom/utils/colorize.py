from colorama import Fore, Style


def color_str(string, color, bold=False):
    r"""Returns stylized string with coloring and bolding for printing.
    
    Example::
    
        >>> print(color_str('lagom', 'green', bold=True))

    See `colorama`_ documentation for more details. 
    
    .. colorama:
        https://pypi.org/project/colorama/
        
    Args:
        string (str): input string
        color (str): color name
        bold (bool, optional): if ``True``, then the string is bolded. Default: ``False``
    
    Returns:
        out: stylized string
    
    """
    colors = {'red': Fore.RED, 'green': Fore.GREEN, 'blue': Fore.BLUE, 'cyan': Fore.CYAN, 
              'magenta': Fore.MAGENTA, 'black': Fore.BLACK, 'white': Fore.WHITE}
    style = colors[color]
    if bold:
        style += Style.BRIGHT
    out = style + string + Style.RESET_ALL
    return out
