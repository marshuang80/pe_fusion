import torch.nn as nn


class FCNN(nn.Module):

    def __init__(
            self, 
            feature_size:int, 
            num_neurons:int, 
            num_hidden:int, 
            init_method:str, 
            activation:str,
            dropout_prob:float = 0, 
            classify:bool=True
        ):

        super(FCNN, self).__init__()

        # input args
        self.feature_size = feature_size
        self.num_neurons = num_neurons
        self.dropout_prob = dropout_prob
        self.init_method = init_method

        # get activation
        self.get_activation(activation)

        # input layer
        self.feed_forward = nn.ModuleList()
        self.feed_forward.extend([nn.Linear(feature_size, self.num_neurons), self.activation()])

        # hidden layers
        for _ in range(num_hidden):
            self.feed_forward.extend([
                nn.Linear(self.num_neurons, self.num_neurons), \
                self.activation(), \
                nn.Dropout(self.dropout_prob)
            ])

        # output layer
        if classify:
            self.feed_forward.extend([nn.Linear(self.num_neurons, 1)])

        # init weight
        if self.init_method is not None:
            self.initialize_weights(self.init_method)

    def initialize_weights(self, init_method, gain=0.2):
        """Initialize all weights in the network."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                else:
                    raise NotImplementedError('Invalid initialization method: {}'.format(init_method))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_activation(self, activation): 
        # get activation
        if activation == "ELU":
            self.activation =  nn.ELU
        elif activation == "LeakyReLU":
            self.activation =  nn.LeakyReLU
        elif activation == "Tanh":
            self.activation =  nn.Tanh
        else:
            self.activation =  nn.ReLU

    def forward(self, x):
        for layer in self.feed_forward:
            x = layer(x)
        return x
