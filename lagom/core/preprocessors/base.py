class BasePreprocessor(object):
    def process(self, x):
        """
        Process the input data
        
        Args:
            x: input data
            
        Returns:
            out: The processed data
        """
        raise NotImplementedError