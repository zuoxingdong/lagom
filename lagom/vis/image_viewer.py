import pyglet

try:
    import pyglet.gl as gl
except Exception:
    msg1 = '1. make sure OpenGL is installed by running `sudo apt install python-opengl`. \n'
    msg2 = '2. if you are on a server, then create a fake screen with xvfb-run and make sure nvidia driver '
    msg3 = 'is installed with --no-opengl-files and cuda with --no-opengl-libs'
    raise ImportError(msg1+msg2+msg3) from None


class ImageViewer(pyglet.window.Window):
    r"""Display an image from an RGB array in an OpenGL window. 
    
    Example::
    
        imageviewer = ImageViewer(max_width=500)
        image = np.asarray(Image.open('x.jpg'))
        imageviewer(image)
        
    """
    def __init__(self, max_width=500):
        r"""Initialize the OpenGL window. 
        
        Args:
            max_width (int): maximum width of the window. 
        """
        self.max_width = max_width
        self.closed = False
        
        # Create OpenGL window
        super().__init__(visible=False, vsync=False, resizable=True)
    
    def __call__(self, x):
        r"""Create an image from the given RGB array and display to the window. 
        
        Args:
            x (ndarray): RGB array
        """
        assert isinstance(x, np.ndarray), f'expected numpy array dtype, got {type(x)}'
        assert x.ndim == 3, f'expected ndim=3, got {x.ndim}'
        assert x.shape[-1] == 3, f'expected 3 color channel, got {x.shape[-1]}'
        
        # Rescale the window according to the image and maximally allowed width
        img_height, img_width, _ = x.shape
        if img_width > self.max_width:  # too large, rescale
            ratio = self.max_width/img_width
            win_width = int(ratio*img_width)
            win_height = int(ratio*img_height)
        else:  # allowed range
            win_width = img_width
            win_height = img_height
        # Resize the window
        self.set_size(width=win_width, height=win_height)
        
        # Set window to be visible for first call
        if not self.visible:
            self.set_visible(True)
        
        # Create an image object
        image = pyglet.image.ImageData(width=img_width, 
                                       height=img_height, 
                                       format='RGB', 
                                       data=x.tobytes(), 
                                       pitch=img_width*-3)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture = image.get_texture()
        # Resize the image texture as the window size
        texture.width = self.width
        texture.height = self.height
        
        # Clear the window and display the image
        self.clear()
        self.switch_to()
        self.dispatch_events()
        texture.blit(0, 0)  # draw image to active framebuffer displaying at lower-left corner
        self.flip()  # filp the front and backend buffer

    def close(self):
        r"""Close the Window. """
        if not self.closed:
            self.closed = True
            super().close()
