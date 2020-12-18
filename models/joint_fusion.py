import torch
import torch.nn as nn

from .fcnn import FCNN

class JointModel(nn.Module):

    def __init__(
            self, 
            feature_size:list, 
            num_neurons:int, 
            dropout_prob:float, 
            init_method:str, 
            num_hidden:int, 
            activation:str
        ):

        super(JointModel, self).__init__()

        # input args
        self.feature_size = feature_size
        self.num_neurons = num_neurons
        self.dropout_prob = dropout_prob
        self.init_method = init_method
        self.num_hidden = num_hidden
        self.activation = activation

        # feature extractor
        self.feature_fcnns = []

        # fcnn for each input modality 
        for size in feature_size:
            feature_fcnn = FCNN(
                feature_size=size, 
                num_neurons=self.num_neurons,
                num_hidden=self.num_hidden,
                init_method=self.init_method,
                activation=self.activation,
                dropout_prob=self.dropout_prob,
                classify=False
            )
            self.feature_fcnns.append(feature_fcnn)
        self.feature_fcnns = nn.ModuleList(self.feature_fcnns)

        # classifier head
        classifier_input_dim = self.num_neurons*len(feature_size)
        self.classifier_layers = FCNN(
                feature_size=classifier_input_dim, 
                num_neurons=self.num_neurons,
                num_hidden=self.num_hidden,
                init_method=self.init_method,
                activation=self.activation,
                dropout_prob=self.dropout_prob
        )

    def forward(self, x):

        # get features from each modality's FCNN
        features = []
        for feature, feature_fcnn in zip(x, self.feature_fcnns):
            feature = feature_fcnn(feature)
            features.append(feature)
        joint_features = torch.cat(features, 1)

        # classifier layers
        pred = self.classifier_layers(joint_features)
        return pred