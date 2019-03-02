import numpy as np

from PIL import Image


class GridImage(object):
    r"""Generate a grid of images. The images can be iteratively added. 
    
    Example::
    
        grid = GridImage(ncol=8, padding=5, pad_value=0)

        a = np.random.randint(0, 255+1, size=[10, 3, 64, 64])
        grid.add(a)
        grid()
    
    Reference:
    
        * https://github.com/pytorch/vision/blob/master/torchvision/utils.py
        
        * https://github.com/facebookresearch/visdom/blob/master/py/visdom/__init__.py
        
    Args:
        ncol (int, optional): Number of images to show in each row of the grid. 
            Final grid size is [N/ncol, ncol]. Default: 8. 
        padding (int, optional): Number of paddings. Default: 2.
        pad_value (float, optional): Padding value in the range [0, 255]. 
            Black is 0 and white 255. Default: 0

    """
    def __init__(self, ncol=8, padding=2, pad_value=0):
        self.ncol = ncol
        self.padding = padding
        self.pad_value = pad_value
        
        # Data buffer
        self.x = None

    def add(self, x):
        r"""Add a new data for making grid images. 
        
        Args:
            x (list/ndarray): a list or ndarray of images, with shape either [H, W], [C, H, W] or [N, C, H, W]
        """
        if not isinstance(x, (list, np.ndarray)):
            raise TypeError(f'list or ndarray expected, got {type(x)}')

        x = np.array(x)
        assert x.ndim <= 4 or x.ndim >= 2, f'either 2, 3, or 4 dimensions expected, got {x.ndim}'

        # Convert to shape [N, C, H, W]
        if x.ndim == 2:  # Single image HxW -> [1, 1, H, W]
            x = x.reshape([1, 1, *x.shape])
        elif x.ndim == 3:  # Single image CxHxW -> [1, C, H, W]
            x = x.reshape([1, *x.shape])

        # Convert to RGB channels for single color channel
        if x.shape[1] == 1:
            x = np.concatenate([x]*3, axis=1)
            
        # Save to data buffer
        if self.x is None:
            self.x = x
        else:  # concatenate with existing images in data buffer, along batch dimension N
            self.x = np.concatenate([self.x, x], axis=0)
        
    def __call__(self, **kwargs):
        r"""Make grid of images. 
        
        Args:
            **kwargs: keyword aguments used to specify the grid of images. 
            
        Returns
        -------
        img : Image
            a grid of image with shape [H, W, C] and dtype ``np.uint8``
        """
        # Total number of images
        N = self.x.shape[0]
        # Number of images in one row
        cols = min(N, self.ncol)
        # Number of rows, at least one
        rows = int(np.ceil(N/cols))
        # Image height
        img_H = self.x.shape[2]
        # Image width
        img_W = self.x.shape[3]
        # Padded height
        H = img_H + self.padding
        # Padded width
        W = img_W + self.padding
        
        # Create a grid
        grid = np.full([3, rows*H + self.padding, cols*W + self.padding], float(self.pad_value))

        n = 0
        for row in range(rows):
            for col in range(cols):
                if n >= N:  # terminate when finish all images
                    break
                H_start = row*H + self.padding
                H_end = H_start + img_H
                W_start = col*W + self.padding
                W_end = W_start + img_W

                # Fill the image
                grid[:, H_start:H_end, W_start:W_end] = self.x[n]

                n += 1

        # Enforce unit8 images in the range [0, 255]
        if 'float' in str(grid.dtype):
            if grid.max() <= 1:  # value range in [0, 1]
                grid *= 255.
            grid = grid.astype(np.uint8)

        # Convert to shape [H, W, C]
        grid = np.transpose(grid, axes=[1, 2, 0])

        img = Image.fromarray(grid)
        
        return img
