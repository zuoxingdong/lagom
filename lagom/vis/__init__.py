try:  # workaround on server without fake screen but still running other things well
    from .image_viewer import ImageViewer
except ImportError:
    import warnings
    warnings.warn('ImageViewer failed to import due to pyglet. ')

from .grid_image import GridImage

from .utils import set_ticker
