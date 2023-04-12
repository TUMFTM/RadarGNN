from radargnn.gnn.mpnn_layers import MPNNConv, RadarPointGNNConv
from radargnn.gnn.gnn_models import get_mlp, DetNetBasic
from radargnn.gnn.configs import GNNArchitectureConfig
import torch
from torch_geometric.nn.dense.linear import Linear
import numpy as np


def test_get_mlp():
    mlp = get_mlp(2, 3, [5], False)

    # set all weights to 1 and biases to 0
    for layer in mlp:
        if isinstance(layer, Linear):
            layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

    x = torch.tensor([1, 1], dtype=torch.float32)

    # check layer dimensions and propagation
    t1 = mlp[0].weight.detach().numpy().shape == (5, 2)
    t2 = mlp[2].weight.detach().numpy().shape == (3, 5)
    t3 = (mlp(x).detach().numpy() == np.array([10, 10, 10, ])).all()

    assert (all((t1, t2, t3)))


def test_det_net_basic_with_mpnn_conv():
    config = GNNArchitectureConfig(
        2, 3, [5], [3], [3], conv_layer_type="MPNNConv")
    model = DetNetBasic(config)
    assert (isinstance(model.convs[0], MPNNConv))


def test_det_net_basic_with_pint_gnn_conv():
    config = GNNArchitectureConfig(
        2, 3, [2], [3], [3], conv_layer_type="RadarPointGNNConv")
    model = DetNetBasic(config)
    assert (isinstance(model.convs[0], RadarPointGNNConv))


def test_radar_point_gnn_conv():
    conv = RadarPointGNNConv(2, 1, "max", 2, 1)

    # set all weights to 1 and biases to 0
    for layer in conv.pre_mlp:
        if isinstance(layer, Linear):
            layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

    # set all weights to 1 and biases to 0
    for layer in conv.post_mlp:
        if isinstance(layer, Linear):
            layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

    # input contains one feature vector and edge feature vector
    pre_mlp_x = torch.tensor([[1, 1, 1],
                              [2, 2, 2]], dtype=torch.float32)

    # input contains pre_mlp output message and node feature vector
    post_mlp_x = torch.tensor([[1, 1, 1, 1, 1],
                               [2, 2, 2, 2, 2]], dtype=torch.float32)

    b1 = False
    try:
        _ = conv.pre_mlp(pre_mlp_x)
        _ = conv.post_mlp(post_mlp_x)
        b1 = True
    except Exception():
        pass

    b2 = len(conv.pre_mlp) == 3
    b3 = len(conv.post_mlp) == 1

    assert (all([b1, b2, b3]))


def test_general_mpnn_conv_mlps():

    conv = MPNNConv(2, 4, 3, post_layers=2)

    # set all weights to 1 and biases to 0
    for layer in conv.pre_mlp:
        if isinstance(layer, Linear):
            layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

    # set all weights to 1 and biases to 0
    for layer in conv.post_mlp:
        if isinstance(layer, Linear):
            layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

    # input contains both node feature vectors and edge feature vector
    pre_mlp_x = torch.tensor([[1, 1, 1, 1, 1, 1, 1],
                              [2, 2, 2, 2, 2, 2, 2]], dtype=torch.float32)
    pre_mlp_y = conv.pre_mlp(pre_mlp_x)

    # input contains pre_mlp output message and node feature vector
    post_mlp_x = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [2, 2, 2, 2, 2, 2, 2, 2, 2]], dtype=torch.float32)
    post_mlp_y = conv.post_mlp(post_mlp_x)

    b1 = len(conv.pre_mlp) == 1
    b2 = len(conv.post_mlp) == 3

    # first mlp output: 7*1
    b3 = all(pre_mlp_y[0, :].detach().numpy() == np.array([7, 7, 7, 7, 7, 7, 7]))

    # second mlp output: 9 * 2 * 4
    # -> first layer: [9*2, 9*2, 9*2, 9*2]
    # -> second layer: [18 * 4, 18 * 4, 18 * 4, 18 * 4])
    b4 = all(post_mlp_y[1, :].detach().numpy() == np.array([72, 72, 72, 72]))

    assert (all([b1, b2, b3, b4]))


def test_general_mpnn_conv_forward():

    conv = MPNNConv(2, 4, 3, post_layers=2, aggr="max")

    # set all weights to 1 and biases to 0
    for layer in conv.pre_mlp:
        if isinstance(layer, Linear):
            layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

    # set all weights to 1 and biases to 0
    for layer in conv.post_mlp:
        if isinstance(layer, Linear):
            layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

    # node features
    x = torch.tensor([[1, 1],
                      [2, 2]], dtype=torch.float32)

    # edges
    edge_index = torch.tensor([[0, 1, 0],
                               [1, 0, 1]], dtype=torch.long)
    # edge features
    edge_attr = torch.tensor([[3, 3, 3],
                              [4, 4, 4],
                              [1, 1, 1]], dtype=torch.float32)

    # forward pass of general MPNN conv layer
    conv_out = conv.forward(x, edge_index, edge_attr)

    # imitate conv_layer calculations
    # updating for x1
    # edges 0 -> 1 relevant ([3,3,3] and [1,1,1])
    # features for x0, x1, e01 (only use e01 with bigger edge -> max aggregation ignores smaller one anyway)
    x_pre_mlp = np.array([1, 1, 2, 2, 3, 3, 3])

    # imitate first mlp with one layer and ones (only for bigger edge [3,3,3])
    y_pre_mlp = np.ones_like(x_pre_mlp) * np.sum(x_pre_mlp)

    # max aggregation only choses the bigger message
    # -> = y_pre_mlp because [3,3,3] > [1,1,1]
    # message = y_pre_mlp

    # get input for second mlp
    # message, x1
    x_post_mlp = np.concatenate((y_pre_mlp, np.array([2, 2])))

    # imitate second mlp with two layers
    # -> has output dim of 4 like defined for new embedding
    y_1_post_mlp = np.array([1, 1, 1, 1]) * np.sum(x_post_mlp)
    y_2_post_mlp = np.array([1, 1, 1, 1]) * np.sum(y_1_post_mlp)

    assert (all((y_2_post_mlp == conv_out[1, :].detach().numpy())))


def test_general_mpnn_conv_edge_encoder():
    conv = MPNNConv(1, 4, 2, use_edge_encoder=True)

    # set all weights to 1 and biases to 0
    for layer in conv.pre_mlp:
        if isinstance(layer, Linear):
            layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

    # set all weights to 1 and biases to 0
    for layer in conv.post_mlp:
        if isinstance(layer, Linear):
            layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
            layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))

    # set all weights of edge encoder to 2 and bias to 0
    conv.edge_encoder.weight = torch.nn.Parameter(
        torch.ones_like(conv.edge_encoder.weight) * 2)
    conv.edge_encoder.bias = torch.nn.Parameter(
        torch.zeros_like(conv.edge_encoder.bias))

    # node features
    x = torch.tensor([[1],
                      [2]], dtype=torch.float32)

    # edges
    edge_index = torch.tensor([[0, 1],
                               [1, 0]], dtype=torch.long)
    # edge features
    edge_attr = torch.tensor([[1, 1],
                              [2, 2]], dtype=torch.float32)

    # forward pass of general MPNN conv layer
    conv_out = conv.forward(x, edge_index, edge_attr)

    b1 = conv.edge_encoder(edge_attr)[0].item() == 4

    # input and output dimension of first MLP is now 3 and not 4
    b2 = conv.pre_mlp[0].weight[0].shape[0] == 3

    # final output for node feature embedding of x1
    # step 1: edge encoding: [1,1] * weights[2,2].T = 4
    # step 2: mlp([1,2,4]) = [7,7,7]
    # step 3: mlp([1,7,7,7]) = [23,23,23,23]
    b3 = conv_out[1, 0].item() == 23

    assert (all([b1, b2, b3]))
