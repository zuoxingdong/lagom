from .base_network import BaseNetwork


class BaseCNN(BaseNetwork):
    """
    Base class for convolutional neural networks. 
    
    Note that if subclass overrides __init__, remember to provide
    keywords aguments, i.e. **kwargs passing to super().__init__. 
    
    All inherited subclasses should at least implement the following functions:
    1. make_params(self, config)
    2. init_params(self, config)
    3. forward(self, x)
    
    Examples:
    
    class CNN(BaseCNN):
        def make_params(self, config):
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3)
            self.fc1 = nn.Linear(in_features=4608, out_features=1)

        def init_params(self, config):
            gain = nn.init.calculate_gain('relu')

            nn.init.orthogonal_(self.conv1.weight, gain=gain)
            nn.init.constant_(self.conv1.bias, 0.0)

            nn.init.orthogonal_(self.conv2.weight, gain=gain)
            nn.init.constant_(self.conv2.bias, 0.0)

            nn.init.orthogonal_(self.fc1.weight, gain=gain)
            nn.init.constant_(self.fc1.bias, 0.0)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.shape[0], -1)
            print(x.shape)
            x = self.fc1(x)

            return x
    """
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
