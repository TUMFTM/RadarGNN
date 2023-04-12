import numpy as np
import gnnradarobjectdetection.graph_constructor.graph as gr
import gnnradarobjectdetection.graph_constructor.features as ft


def test_point_pair_features():
    p1 = np.array([1, 1]).reshape(2, 1)
    p2 = np.array([3, 2]).reshape(2, 1)
    v1 = np.array([0, 1]).reshape(2, 1)
    v2 = np.array([1, 0]).reshape(2, 1)

    d, theta_v1_v2, theta_d_v1, theta_d_v2 = ft.get_En_equivariant_point_pair_metrics(
        p1, p2, v1, v2, "directed")
    res = [round(d, 2), round(theta_v1_v2, 2), round(
        theta_d_v1, 2), round(theta_d_v2, 2)]
    res_true = [2.24, 90.0, 63.43, 26.57]
    assert (res == res_true)


def test_point_pair_features_with_zero_velocity():
    p1 = np.array([1, 1]).reshape(2, 1)
    p2 = np.array([3, 2]).reshape(2, 1)
    v1 = np.array([0, 1]).reshape(2, 1)
    v2 = np.array([0, 0]).reshape(2, 1)

    d, theta_v1_v2, theta_d_v1, theta_d_v2 = ft.get_En_equivariant_point_pair_metrics(
        p1, p2, v1, v2, "directed")
    res = [round(d, 2), round(theta_v1_v2, 2), round(
        theta_d_v1, 2), round(theta_d_v2, 2)]
    res_true = [2.24, 90.0, 63.43, 90.0]
    assert (res == res_true)


def test_edge_features():
    X = np.array([[1, 1],
                  [3, 2]])

    V = np.array([[0, 1],
                  [1, 0]])

    rcs = np.array([0,
                    1]).reshape(2, 1)

    graph = gr.GeometricGraph()
    graph.X = X
    graph.V = V
    graph.F = {"rcs": rcs}
    graph.build(X, "knn", k=1)

    # extract different edge features
    features = ["point_pair_features", "spatial_euclidean_distance",
                "velocity_euclidean_distance", "relative_position", "relative_velocity"]
    edge_mode = "directed"
    graph.extract_node_pair_features(features, edge_mode)

    # features for edge from x0 -> x1: must contain relative position from x0 w.r.t x1
    res_true = [2.24, 90, 63.43, 26.57, 2.24, 1.41, -2, -1, -1, 1]

    assert (res_true == np.round(graph.E_feat[0, :], 2).tolist())


def test_node_features():
    X = np.array([[1, 1],
                  [3, 2]])

    V = np.array([[0, 1],
                  [1, 0]])

    rcs = np.array([1.8,
                    2.6]).reshape(2, 1)

    time_index = np.array([100,
                          101]).reshape(2, 1)

    graph = gr.GeometricGraph()
    graph.X = X  # position of node
    graph.V = V  # velocity of node
    # invariant features of node
    graph.F = {"rcs": rcs, "time_index": time_index}
    graph.build(X, "knn", k=1)

    features = ["rcs", "time_index", "degree", "velocity_vector_length",
                "velocity_vector", "spatial_coordinates"]
    graph.extract_single_node_features(features)

    res_true = [2.6, 101, 1, 1, 1, 0, 3, 2]

    assert (res_true == graph.X_feat[1, :].tolist())


def test_add_degree_to_inv_features():
    X = np.array([[1, 1],
                  [3, 2]])

    graph = gr.GeometricGraph()
    graph.build(X, "knn", k=1)

    # adding degree as first inv features F
    graph.add_degree_to_inv_features()
    # add degree to already exiting inv features F
    graph.add_degree_to_inv_features()

    assert (np.sum(graph.F.get("degree") == np.array([[1, 1], [1, 1]])) == 4)


def test_add_node_feature():
    X = np.array([[1, 1],
                  [3, 2]]).reshape(2, 2)

    rcs = np.array([-1,
                    -2]).reshape(2, 1)

    graph = gr.GeometricGraph()
    graph.X = X
    graph.F = {"rcs": rcs}

    graph.build(X, "knn", k=1)
    graph.add_node_features(rcs)
    graph.extract_single_node_features(["rcs"])
    graph.add_node_features(rcs)

    assert ((graph.X_feat[0, :] == [-1, -1, -1]).all())
