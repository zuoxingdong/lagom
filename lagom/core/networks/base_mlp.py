from .base_network import BaseNetwork


class BaseMLP(BaseNetwork):
    """
    Base class for fully connected neural network (or Multi-Layer Perceptron)
    
    All inherited subclass should implement the following functions
    1. make_params(self, config)
    2. init_params(self, config)
    3. forward(self, x)
    
    Examples: 
    
    class MLP(BaseMLP):
        def make_params(self, config):
            self.fc1 = nn.Linear(in_features=5, out_features=32)
            self.fc2 = nn.Linear(in_features=32, out_features=10)

        def init_params(self, config):
            gain = nn.init.calculate_gain(nonlinearity='relu')

            nn.init.orthogonal_(self.fc1.weight, gain=gain)
            nn.init.constant_(self.fc1.bias, 0.0)

            nn.init.orthogonal_(self.fc2.weight, gain=gain)
            nn.init.constant_(self.fc2.bias, 0.0)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

            return x
    """
    def __init__(self, config=None):
        super().__init__(config)
