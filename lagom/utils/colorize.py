import colored
from colored import stylize


def color_str(string, color, attribute=None):
    r"""Returns stylized string with color and attribute for printing. 
    
    Example::
    
        >>> print(color_str('lagom', 'green', attribute='bold'))
    
    See `colored`_ documentation for more details. 
    
    .. _colored:
        https://pypi.org/project/colored
    
    Args:
        string (str): input string
        color (str): color name
        attribute (str, optional): attribute. Default: ``None``
    
    Returns:
        str: stylized string
    """
    styles = colored.fg(color)
    if attribute is not None:
        styles += colored.attr(attribute)
        
    out = stylize(string, styles)
    
    return out
