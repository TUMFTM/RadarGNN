
from typing import List

import torch
from torch_geometric.nn.dense.linear import Linear
from torch.nn import ReLU, Sequential, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm

from radargnn.gnn.mpnn_layers import MPNNConv, RadarPointGNNConv
from radargnn.gnn.configs import GNNArchitectureConfig


# GNN for object detection
class DetNetBasic(torch.nn.Module):
    """ GNN for end to end object detection and semantic segmentation.

    The model consists of an initial node and edge feature embedding, followed by graph convolution layers,
    and a final detection head for classification and bounding box prediction for each point.
    """

    def __init__(self, config: GNNArchitectureConfig):

        super().__init__()

        # Store the settings of the model as instance attributes for later access
        self.batch_norm_mlps = config.batch_norm_in_mlps

        self.node_feat_dim = config.node_feature_dimension
        self.edge_feat_dim = config.edge_feature_dimension

        self.conv_layer_dimensions = config.conv_layer_dimensions
        self.initial_node_feature_embedding = config.initial_node_feature_embedding
        self.initial_edge_feature_embedding = config.initial_edge_feature_embedding

        self.conv_pre_mlp_layers = config.conv_pre_mlp_layer_number
        self.conv_post_mlp_layers = config.conv_post_mlp_layer_number
        self.conv_use_edge_encoder = config.conv_use_edge_encoder
        self.aggregation = config.aggregation_function

        # Create the MLPs for initial node and edge feature embedding
        if config.initial_node_feature_embedding:
            layer_dimensions = config.node_feature_embedding_layer_dimensions[:-1]
            out_dim = config.node_feature_embedding_layer_dimensions[-1]
            self.node_emb_mlp = get_mlp(self.node_feat_dim, out_dim, layer_dimensions, self.batch_norm_mlps)
            self.node_feat_dim = out_dim

        if config.initial_edge_feature_embedding:
            layer_dimensions = config.edge_feature_embedding_layer_dimensions[:-1]
            out_dim = config.edge_feature_embedding_layer_dimensions[-1]
            self.edge_emb_mlp = get_mlp(self.edge_feat_dim, out_dim, layer_dimensions, self.batch_norm_mlps)
            self.edge_feat_dim = out_dim

        # graph convolutional layer definition - create the first graph convolution layer
        # MPNNConv layer should be interchangeable by any other Graph Convolution layer operating on node and edge features (e.g. GATConv, PNAConv)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        layer_dim = self.conv_layer_dimensions[0]
        if config.conv_layer_type == "MPNNConv":
            conv = MPNNConv(self.node_feat_dim, layer_dim, self.edge_feat_dim, aggr=self.aggregation,
                            pre_layers=self.conv_pre_mlp_layers, post_layers=self.conv_post_mlp_layers, use_edge_encoder=self.conv_use_edge_encoder)
        elif config.conv_layer_type == "RadarPointGNNConv":
            conv = RadarPointGNNConv(self.node_feat_dim, self.edge_feat_dim, aggr=self.aggregation,
                                     pre_layers=self.conv_pre_mlp_layers, post_layers=self.conv_post_mlp_layers)
            layer_dim = self.node_feat_dim
        else:
            raise Exception(
                f"{config.conv_layer_type} is invalid GNN conv layer type. Chose either MPNNConv or RadarPointGNNConv")

        batch_norm = BatchNorm(layer_dim)
        self.convs.append(conv)
        self.batch_norms.append(batch_norm)

        # graph convolutional layer definition - create the remaining graph convolution layers
        for next_layer_dim in self.conv_layer_dimensions[1:]:
            if config.conv_layer_type == "MPNNConv":
                conv = MPNNConv(layer_dim, next_layer_dim, self.edge_feat_dim, aggr=self.aggregation,
                                pre_layers=self.conv_pre_mlp_layers, post_layers=self.conv_post_mlp_layers, use_edge_encoder=self.conv_use_edge_encoder)
            elif config.conv_layer_type == "RadarPointGNNConv":
                conv = RadarPointGNNConv(self.node_feat_dim, self.edge_feat_dim, aggr=self.aggregation,
                                         pre_layers=self.conv_pre_mlp_layers, post_layers=self.conv_post_mlp_layers)
            else:
                raise Exception(f"{config.conv_layer_type} is invalid GNN conv layer type. Chose either MPNNConv or RadarPointGNNConv")

            batch_norm = BatchNorm(next_layer_dim)
            self.convs.append(conv)
            self.batch_norms.append(batch_norm)
            layer_dim = next_layer_dim

        # Define detection head with classification and bounding box regression MLPs
        final_embedding_dim = self.conv_layer_dimensions[-1]

        layer_dimensions = config.classification_head_layer_dimensions[:-1]
        out_dim = config.classification_head_layer_dimensions[-1]
        self.classification_head = get_mlp(final_embedding_dim, out_dim, layer_dimensions, self.batch_norm_mlps)
        # Classification head has no softmax/logsoftmax in the end as this is integrated in Pytorch into the CrossEntropyLoss function (which is called during training for loss calculation)
        # self.classification_head.append(LogSoftmax(dim = 1))

        layer_dimensions = config.regression_head_layer_dimensions[:-1]
        out_dim = config.regression_head_layer_dimensions[-1]
        self.regression_head = get_mlp(final_embedding_dim, out_dim, layer_dimensions, self.batch_norm_mlps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """ Forward pass of the model.

        Args:
            x: Node feature matrix.
            edge_index: Edge connection matrix (Alternative representation of sparse adjacency matrix)
            edge_attr: Edge feature matrix.

        Returns:
            c: Probability distribution for the class prediction for each node.
            bb: Regressed bounding box for each node.
        """
        # execute MLPs for initial node and edge feature embedding
        if self.initial_node_feature_embedding:
            x = self.node_emb_mlp(x)

        if self.initial_edge_feature_embedding:
            edge_attr = self.edge_emb_mlp(edge_attr)

        # apply graph convolutions followed and nonlinearity
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            # maybe add drop out
            x = F.relu(x)

        # use final node feature embeddings for prediction
        c = self.classification_head(x)
        bb = self.regression_head(x)

        return c, bb


def get_mlp(in_size: int, out_size: int, hidden_layer_sizes: List[int], batch_norm: bool) -> Sequential:
    """ Creates a MLP with the specified layer number and dimension.

    Args:
        in_size: Input layer dimension.
        out_size: Output layer dimension.
        hidden_layer_sizes: Dimensions of the hidden layers.
        batch_norm: Boolean, whether batch norm should be applied between all layers.

    Returns:
        mlp: pytorch sequential object with the created MLP
    """

    if len(hidden_layer_sizes) == 0:
        modules = [Linear(in_size, out_size)]
    else:

        modules = [Linear(in_size, hidden_layer_sizes[0])]
        in_size = hidden_layer_sizes[0]

        if len(hidden_layer_sizes) == 1:
            layer_size = hidden_layer_sizes[0]
        else:
            for layer_size in hidden_layer_sizes[1:]:
                if batch_norm:
                    modules += [BatchNorm(in_size)]
                    # maybe add dropout
                    # modules += [Dropout()]
                modules += [ReLU()]
                modules += [Linear(in_size, layer_size)]
                in_size = layer_size

        if batch_norm:
            modules += [BatchNorm(layer_size)]
            # maybe add dropout
            # modules += [Dropout()]
        modules += [ReLU()]
        modules += [Linear(layer_size, out_size)]

    mlp = Sequential(*modules)

    return mlp
