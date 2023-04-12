
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch.nn import ReLU, Sequential
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.inits import reset


class MPNNConv(MessagePassing):
    """ Implementation of a general MPNN layer with edge features.

    This layer is not included in pytorch_grometric.
    The two layers from pytorch_geometric that are the closest related to this layer are:
        - NNConv (from original MPNN paper, but does not include edge features)
        - PNAConv (Includes general MPNN update with edge features but also advanced aggregators, ...)

    Attributes:
        in_channels: Dimension of initial node features.
        out_channels: Dimension of the embedded node features after the graph convolution layer.
        edge_dim: Dimension of initial edge features.
        use_edge_encoder: Decides whether to use an edge_encoder or not.
        edge_encoder: MLP used for encoding the edge features before the graph convolution.
        pre_mlp: Message function MLP.
        post_mlp: Update function MLP.

    Methods:
        reset_parameters: Resets the parameters of the model.
        forward: Updates node feature vectors.
        message: Calculates the messages.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, aggr: str = "max",
                 pre_layers: int = 1, post_layers: int = 1, use_edge_encoder: bool = False):

        """
        Args:
            in_channels: integer describing the dimension of initial node features
            out_channels: integer describing the dimension of node feature embeddings returned by this layer
            edge_dim: integer describing the dimension of initial edge features
            aggr (optional): permutation invariant aggregation function
            pre_layers (optional): number of layers in message MLP of the graph convolution operation
            post_layers (optional): number of layers in update MLP of the graph convolution operation
            use_edge_encoder (optional): boolean to chose weather a MLP should be used to transform the edge features to the same dimension as the node features
        """

        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.use_edge_encoder = use_edge_encoder

        # can be used to adapt edge feature dimension to be equal to node feature dimension -> Done this way in PNAConv
        if use_edge_encoder:
            self.edge_encoder = Linear(edge_dim, self.in_channels)
            pre_mlp_dim = 3 * in_channels
        else:
            pre_mlp_dim = 2 * in_channels + edge_dim

        # MLPs before and after aggregation
        # or maybe better reduce output dim of message MLP to match node feature dim ?! -> like in PNAConv
        modules = [Linear(pre_mlp_dim, pre_mlp_dim)]
        for _ in range(pre_layers - 1):
            modules += [ReLU()]
            modules += [Linear(pre_mlp_dim, pre_mlp_dim)]
        self.pre_mlp = Sequential(*modules)

        modules = [Linear(pre_mlp_dim + in_channels, out_channels)]
        for _ in range(post_layers - 1):
            modules += [ReLU()]
            modules += [Linear(out_channels, out_channels)]
        self.post_mlp = Sequential(*modules)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_edge_encoder:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_mlp:
            reset(nn)
        for nn in self.post_mlp:
            reset(nn)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:

        m_emb = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.cat([x, m_emb], dim=-1)
        h = self.post_mlp(out)

        return h

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:

        if self.use_edge_encoder:
            edge_attr = self.edge_encoder(edge_attr)
        m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        m_emb = self.pre_mlp(m)

        return m_emb


class RadarPointGNNConv(MessagePassing):
    """ Adapted Radar-PointGNN convolution with edge features.

    Graph convolution layer as specified in: Radar-PointGNN: Graph Based Object Recognition for Unstructured Radar Point-cloud Data.
    (DOI: 10.1109/RadarConf2147009.2021.9455172).
    BUT with the following changes:
    instead of only using "x_j - x_i, m_j" in the pre_mlp, all edge attributes are used which may contain also other features than the relative position "e_ij, m_j"

    Attributes:
        in_channels: Dimension of initial node features.
        out_channels: Dimension of the embedded node features after the graph convolution layer.
        init_node_dim: Dimension of initial node features.
        init_edge_dim: Dimension of initial edge features.
        pre_mlp: Message function MLP.
        post_mlp: Update function MLP.

    Methods:
        reset_parameters: Resets the parameters of the model.
        forward: Updates node feature vectors.
        message: Calculates the messages.
    """

    def __init__(self, init_node_dim: int, init_edge_dim: int, aggr: str = "max",
                 pre_layers: int = 1, post_layers: int = 1):

        """
        Args:
            init_node_dim: integer describing the dimension of initial node features
            init_edge_dim: integer describing the dimension of node feature embeddings returned by this layer
            aggr (optional): permutation invariant aggregation function
            pre_layers (optional): number of layers in message MLP of the graph convolution operation
            post_layers (optional): number of layers in update MLP of the graph convolution operation
        """
        super().__init__(aggr=aggr)

        # output dim. = input dim. -> No increase in embedding dimension possible with this layer
        self.in_channels = init_node_dim
        # output dim. = input dim. -> No increase in embedding dimension possible with this layer
        self.out_channels = init_node_dim

        self.init_node_dim = init_node_dim
        self.init_edge_dim = init_edge_dim

        pre_mlp_dim = init_node_dim + init_edge_dim

        # MLPs before and after aggregation
        # or maybe better reduce output dim of message MLP to match node feature dim ?! -> like in PNAConv
        modules = [Linear(pre_mlp_dim, pre_mlp_dim)]
        for _ in range(pre_layers - 1):
            modules += [ReLU()]
            modules += [Linear(pre_mlp_dim, pre_mlp_dim)]
        self.pre_mlp = Sequential(*modules)

        modules = [Linear(pre_mlp_dim + init_node_dim, init_node_dim)]
        for _ in range(post_layers - 1):
            modules += [ReLU()]
            modules += [Linear(init_node_dim, init_node_dim)]
        self.post_mlp = Sequential(*modules)

        self.reset_parameters()

    def reset_parameters(self):
        for nn in self.pre_mlp:
            reset(nn)
        for nn in self.post_mlp:
            reset(nn)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:

        m_emb = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.cat([x, m_emb], dim=-1)
        h = self.post_mlp(out)

        return h + x

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:

        m = torch.cat([x_j, edge_attr], dim=-1)
        m_emb = self.pre_mlp(m)

        return m_emb
